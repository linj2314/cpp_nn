#include "main.hpp"

using namespace std;
using namespace cv;

//HYPER PARAMETERS---------------------------------------
//batch size; make sure 8000 is divisible by this number
const int BATCH_SIZE = 64;

//learning rate; step size of descending the gradient
const int LEARNING_RATE = 0.2;
//-------------------------------------------------------

//an 8000 x {1, 784} data structure to hold our input data
vector<int> Y_train;
vector<vector<double>> X_train;

//weights and biases vectors
vector<vector<double>> W_1(784, vector<double>(256, 0));
vector<vector<double>> W_2(256, vector<double>(128, 0));
vector<vector<double>> W_3(128, vector<double>(10, 0));
vector<double> B_1(256, 0);
vector<double> B_2(128, 0);
vector<double> B_3(10, 0);

//hidden and output layers
vector<vector<double>> layer1(BATCH_SIZE, vector<double>(256, 0));
vector<vector<double>> layer2(BATCH_SIZE, vector<double>(128, 0));
vector<vector<double>> layer3(BATCH_SIZE, vector<double>(10, 0));

void init() {
    //reading in all pixel data from training jpg files
    vector<string> simpson_characters = {"bart_simpson", 
                                        "charles_montgomery_burns", 
                                        "homer_simpson", 
                                        "krusty_the_clown", 
                                        "lisa_simpson", 
                                        "marge_simpson", 
                                        "milhouse_van_houten", 
                                        "moe_szyslak", 
                                        "ned_flanders", 
                                        "principal_skinner"};

    Y_train.reserve(8000);
    X_train.reserve(8000);
    for (int i = 0; i < 10; i++) {
        string cmd = "cd train && cd " + simpson_characters[i] + " && ls > ../../temp.txt";
        system(cmd.c_str());

        ifstream in = ifstream("temp.txt");
        string file_name;
        while (in >> file_name) {
            vector<double> v(784, 0);
            Mat image = imread("train/" + simpson_characters[i] + "/" + file_name, cv::IMREAD_GRAYSCALE);
            
            for (int j = 0; j < image.rows; j++) {
                for (int k = 0; k < image.cols; k++) {
                    v[j * 28 + k] = static_cast<int>(image.at<uchar>(j, k)) / 255.0;
                }
            }

            Y_train.push_back(i);
            X_train.push_back(v);
        }
    }

    //shuffle training data so we can separate into batches
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    mt19937 rng(seed);

    vector<int> indexes(8000);
    iota(indexes.begin(), indexes.end(), 0);
    shuffle(indexes.begin(), indexes.end(), rng);

    vector<int> Y_train_temp = Y_train;
    vector<vector<double> *> X_train_temp(8000);

    for (int i = 0; i < 8000; i++) {
        Y_train_temp[indexes[i]] = Y_train[i];
        X_train_temp[indexes[i]] = &X_train[i];
    }

    Y_train = move(Y_train_temp);
    for (int i = 0; i < 8000; i++) {
        X_train[i] = *X_train_temp[i];
    }

    //initializing weights
    //"He Weight Initialization" named after Kaiming He
    //randomly generates values according to a normal distribution with mean 0 and sd of sqrt(2/n)
    double sd = sqrt(2.0/784);
    normal_distribution<double> dist(0, sd);

    auto rand_gen = [&dist, &rng](){
        return dist(rng);
    };

    for (auto & i : W_1)
        generate(i.begin(), i.end(), rand_gen);
    for (auto & i : W_2)
        generate(i.begin(), i.end(), rand_gen);
    for (auto & i : W_3)
        generate(i.begin(), i.end(), rand_gen);
};

//forwards propagation function; takes a batch as input and returns the errors
void forward_propagate(vector<vector<double>> & batch) {
    //layer 1
    for (int i = 0; i < BATCH_SIZE; i++) {
        for (int j = 0; j < 256; j++) {
            double dot_product = 0;
            for (int k = 0; k < 728; k++) {
                dot_product += batch[i][k] * W_1[k][j];
            }
            dot_product += B_1[j];

            //apply ReLU to the final dot product for activation
            dot_product = max(dot_product, 0.0);
            layer1[i][j] = dot_product;
        }
    }
    
    //layer 2
    for (int i = 0; i < BATCH_SIZE; i++) {
        for (int j = 0; j < 128; j++) {
            double dot_product = 0;
            for (int k = 0; k < 256; k++) {
                dot_product += layer1[i][k] * W_2[k][j];
            }
            dot_product += B_2[j];

            //apply ReLU to the final dot product for activation
            dot_product = max(dot_product, 0.0);
            layer2[i][j] = dot_product;
        }
    }

    //layer 3
    for (int i = 0; i < BATCH_SIZE; i++) {
        //sum tracking for softmax activation at end;
        double max_val = -DBL_MAX;
        for (int j = 0; j < 10; j++) {
            double dot_product = 0;
            for (int k = 0; k < 128; k++) {
                dot_product += layer2[i][k] * W_3[k][j];
            }
            dot_product += B_3[j];

            max_val = max(max_val, dot_product);
            layer3[i][j] = dot_product;
        }
        //apply softmax in order to get final predictions
        double sum = 0;
        for (int j = 0; j < 10; j++) {
            layer3[i][j] = exp(layer3[i][j] - max_val);
            sum += layer3[i][j];
        }
        for (int j = 0; j < 10; j++) {
            layer3[i][j] /= sum;
        }
    }
}

//backwards propagation function; nudges weights and biases to minimize error using gradient descent
void back_propagate(vector<int> & labels) {
    //calculating current accuracy for display
    double score = 0;
    for (int i = 0; i < BATCH_SIZE; i++) {
        int predicted = -1;
        int highest = INT_MIN;
        for (int j = 0; j < 10; j++) {
            if (layer3[i][j] > highest) {
                predicted = j;
                highest = layer3[i][j];
            }
        }
        if (labels[i] == predicted) {
            score += 1.0;
        }
    }
    cout << "Current Accuracy: " << score / BATCH_SIZE << "\n";

    //calculating errors using the cross entropy loss function
    vector<double> errors(BATCH_SIZE);
    vector<vector<double>> one_hot_encodings(BATCH_SIZE, vector<double>(10, 0));

    for (int i = 0; i < BATCH_SIZE; i++) {
        one_hot_encodings[i][labels[i]] = 1;
        errors[i] = -1.0 * log(layer3[i][labels[i]]);
    }

    //Use gradient descent to adjust parameters, starting from the back layers 
    //adjusting W_3 and B_3
    {
        vector<vector<double>> W_3_gradient(128, vector<double>(10, 0));
        vector<double> B_3_gradient(10, 0);
        for (int i = 0; i < BATCH_SIZE; i++) {
            for (int j = 0; j < 10; j++) {
                B_3_gradient[j] += layer3[i][j] - ((j == labels[i]) ? 1 : 0);
                for (int k = 0; k < 128; k++) {
                    W_3_gradient[k][j] += layer2[i][k] * (layer3[i][j] - ((j == labels[i]) ? 1 : 0));
                }
            }
        }

        //Average out and apply the gradients * learning rate
        for (int i = 0; i < 10; i++) {
            B_3[i] -= LEARNING_RATE * B_3_gradient[i] / BATCH_SIZE;
            for (int j = 0; j < 128; j++) {
                W_3[j][i] -= LEARNING_RATE * W_3_gradient[j][i] / BATCH_SIZE;
            }
        }
    }

    //adjusting W_2 and B_2
    vector<vector<double>> A_2_gradient(BATCH_SIZE, vector<double>(128, 0));
    {
        vector<vector<double>> W_2_gradient(256, vector<double>(128, 0));
        vector<double> B_2_gradient(128, 0);
        for (int i = 0; i < BATCH_SIZE; i++) {
            for (int j = 0; j < 128; j++) {
                int sum = 0;
                for (int k = 0; k < 256; k++) {
                    for (int l = 0; l < 10; l++) {
                        sum += W_3[j][l] * (layer3[i][l] - ((l == labels[i]) ? 1 : 0));
                    }
                    if (layer2[i][j] != 0) {
                        W_2_gradient[k][j] += layer1[i][k] * sum;
                    }
                }
                if (layer2[i][j] != 0) {
                    B_2_gradient[j] += sum;
                }
                A_2_gradient[i][j] = sum;
            }
        }

        //Average out and apply the gradients * learning rate
        for (int i = 0; i < 128; i++) {
            B_2[i] -= LEARNING_RATE * B_2_gradient[i] / BATCH_SIZE;
            for (int j = 0; j < 256; j++) {
                W_2[j][i] -= LEARNING_RATE * W_2_gradient[j][i] / BATCH_SIZE;
            }
        }
    }

    //adjusting W_1 and B_1
    {
        vector<vector<double>> W_1_gradient(784, vector<double>(256, 0));
        vector<double> B_1_gradient(256, 0);
        for (int i = 0; i < BATCH_SIZE; i++) {
            for (int j = 0; j < 256; j++) {
                int sum = 0;
                for (int k = 0; k < 784; k++) {
                    for (int l = 0; l < 128; l++) {
                        if (layer2[i][l] != 0) {
                            sum += W_2[j][l] * A_2_gradient[i][l];
                        }
                    }
                    if (layer1[i][j] != 0) {
                        W_1_gradient[k][j] += layer1[i][k] * sum;
                    }
                }
                if (layer2[i][j] != 0) {
                    B_1_gradient[j] += sum;
                }
            }
        }

        //Average out and apply the gradients * learning rate
        for (int i = 0; i < 256; i++) {
            B_1[i] -= LEARNING_RATE * B_1_gradient[i] / BATCH_SIZE;
            for (int j = 0; j < 784; j++) {
                W_1[j][i] -= LEARNING_RATE * W_1_gradient[j][i] / BATCH_SIZE;
            }
        }
    }
}

void print_trained_parameters() {

}

void train() {
    int batches = 8000 / BATCH_SIZE;
    //TODO replace the 1 with batches later
    for (int i = 0; i < 80; i++) {
        vector<vector<double>> batch(X_train.begin() + i * BATCH_SIZE, X_train.begin() + i * BATCH_SIZE + BATCH_SIZE);
        vector<int> labels(Y_train.begin() + i * BATCH_SIZE, Y_train.begin() + i * BATCH_SIZE + BATCH_SIZE);
        forward_propagate(batch);
        back_propagate(labels);
    }
    print_trained_parameters();
}

int main() {
    //a function to read in training data and initialize weights and biases to random values
    init();

    //the training function
    train();

    return 0;
}