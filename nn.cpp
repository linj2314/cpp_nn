#include "main.hpp"

using namespace std;
using namespace cv;

//HYPER PARAMETERS---------------------------------------
//batch size; make sure 8000 is divisible by this number
const double BATCH_SIZE = 64;

//learning rate; step size of descending the gradient
double LEARNING_RATE = 0.01;
//-------------------------------------------------------

//an 8000 x {1, 784} data structure to hold our input data
vector<int> Y_train;
vector<vector<double>> X_train;

vector<int> Y_val;
vector<vector<double>> X_val;

vector<int> Y_test;
vector<vector<double>> X_test;

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

vector<vector<double>> layer1_val(1000, vector<double>(256, 0));
vector<vector<double>> layer2_val(1000, vector<double>(128, 0));
vector<vector<double>> layer3_val(1000, vector<double>(10, 0));

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
        auto s = system(cmd.c_str());

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

    Y_val.reserve(2000);
    X_val.reserve(2000);
    for (int i = 0; i < 10; i++) {
        string cmd = "cd test && cd " + simpson_characters[i] + " && ls > ../../temp.txt";
        auto s = system(cmd.c_str());

        ifstream in = ifstream("temp.txt");
        string file_name;
        while (in >> file_name) {
            vector<double> v(784, 0);
            Mat image = imread("test/" + simpson_characters[i] + "/" + file_name, cv::IMREAD_GRAYSCALE);
            
            for (int j = 0; j < image.rows; j++) {
                for (int k = 0; k < image.cols; k++) {
                    v[j * 28 + k] = static_cast<int>(image.at<uchar>(j, k)) / 255.0;
                }
            }

            Y_val.push_back(i);
            X_val.push_back(v);
        }
    }

    //shuffle all data
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    mt19937 rng(seed);

    vector<int> indexes(2000);
    iota(indexes.begin(), indexes.end(), 0);
    shuffle(indexes.begin(), indexes.end(), rng);

    vector<int> Y_val_temp = Y_val;
    vector<vector<double>> X_val_temp(2000);

    for (int i = 0; i < 2000; i++) {
        Y_val_temp[i] = Y_val[indexes[i]];
        X_val_temp[i] = X_val[indexes[i]];
    }

    Y_val = move(Y_val_temp);
    X_val = move(X_val_temp);

    X_test = vector<vector<double>>(X_val.begin() + 1000, X_val.end());
    Y_test = vector<int>(Y_val.begin() + 1000, Y_val.end());

    Y_val.resize(1000);
    X_val.resize(1000);

    vector<int> indexes2(8000);
    iota(indexes2.begin(), indexes2.end(), 0);
    shuffle(indexes2.begin(), indexes2.end(), rng);

    vector<int> Y_train_temp(8000);
    vector<vector<double>> X_train_temp(8000, vector<double>(X_train[0].size()));

    for (int i = 0; i < 8000; i++) {
        Y_train_temp[i] = Y_train[indexes2[i]];
        X_train_temp[i] = X_train[indexes2[i]];
    }

    Y_train = move(Y_train_temp);
    X_train = move(X_train_temp);

    //initializing weights
    //"He Weight Initialization" named after Kaiming He
    //randomly generates values according to a normal distribution with mean 0 and sd of sqrt(2/n)
    double sd1 = sqrt(2.0/784);
    double sd2 = sqrt(2.0/256);
    double sd3 = sqrt(2.0/(128 + 10));
    normal_distribution<double> dist1(0, sd1);
    normal_distribution<double> dist2(0, sd2);
    normal_distribution<double> dist3(0, sd3);

    for (auto & i : W_1) {
        for (auto & j : i) {
            j = dist1(rng);
        }
    }
    for (auto & i : W_2) {
        for (auto & j : i) {
            j = dist2(rng);
        }
    }
    for (auto & i : W_3) {
        for (auto & j : i) {
            j = dist3(rng);
        }
    }
};

//forwards propagation function; takes a batch as input and returns the errors
void forward_propagate(vector<vector<double>> & batch) {
    //layer 1
    for (int i = 0; i < BATCH_SIZE; i++) {
        for (int j = 0; j < 256; j++) {
            double dot_product = 0;
            for (int k = 0; k < 784; k++) {
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
void back_propagate(const vector<vector<double>>& batch, const vector<int>& labels) {
    vector<vector<double>> W_3_gradient(128, vector<double>(10, 0));
    vector<double> B_3_gradient(10, 0);
    vector<vector<double>> W_2_gradient(256, vector<double>(128, 0));
    vector<double> B_2_gradient(128, 0);
    vector<vector<double>> W_1_gradient(784, vector<double>(256, 0));
    vector<double> B_1_gradient(256, 0);

    vector<vector<double>> delta3(BATCH_SIZE, vector<double>(10, 0));
    vector<vector<double>> delta2(BATCH_SIZE, vector<double>(128, 0));
    vector<vector<double>> delta1(BATCH_SIZE, vector<double>(256, 0));

    // Calculate output layer error (delta3)
    for (int i = 0; i < BATCH_SIZE; i++) {
        for (int j = 0; j < 10; j++) {
            delta3[i][j] = layer3[i][j] - (j == labels[i] ? 1.0 : 0.0);
        }
    }

    // Backpropagate to layer 2
    for (int i = 0; i < BATCH_SIZE; i++) {
        for (int j = 0; j < 128; j++) {
            double error = 0;
            for (int k = 0; k < 10; k++) {
                error += W_3[j][k] * delta3[i][k];
            }
            delta2[i][j] = error * (layer2[i][j] > 0 ? 1 : 0);  // ReLU derivative
        }
    }

    // Backpropagate to layer 1
    for (int i = 0; i < BATCH_SIZE; i++) {
        for (int j = 0; j < 256; j++) {
            double error = 0;
            for (int k = 0; k < 128; k++) {
                error += W_2[j][k] * delta2[i][k];
            }
            delta1[i][j] = error * (layer1[i][j] > 0 ? 1 : 0);  // ReLU derivative
        }
    }

    // Calculate gradients
    for (int i = 0; i < BATCH_SIZE; i++) {
        // Layer 3 gradients
        for (int j = 0; j < 10; j++) {
            B_3_gradient[j] += delta3[i][j];
            for (int k = 0; k < 128; k++) {
                W_3_gradient[k][j] += layer2[i][k] * delta3[i][j];
            }
        }

        // Layer 2 gradients
        for (int j = 0; j < 128; j++) {
            B_2_gradient[j] += delta2[i][j];
            for (int k = 0; k < 256; k++) {
                W_2_gradient[k][j] += layer1[i][k] * delta2[i][j];
            }
        }

        // Layer 1 gradients
        for (int j = 0; j < 256; j++) {
            B_1_gradient[j] += delta1[i][j];
            for (int k = 0; k < 784; k++) {
                W_1_gradient[k][j] += batch[i][k] * delta1[i][j];
            }
        }
    }

    // Update weights and biases
    for (int i = 0; i < 10; i++) {
        B_3[i] -= LEARNING_RATE * B_3_gradient[i] / BATCH_SIZE;
        for (int j = 0; j < 128; j++) {
            W_3[j][i] -= LEARNING_RATE * W_3_gradient[j][i] / BATCH_SIZE;
        }
    }

    for (int i = 0; i < 128; i++) {
        B_2[i] -= LEARNING_RATE * B_2_gradient[i] / BATCH_SIZE;
        for (int j = 0; j < 256; j++) {
            W_2[j][i] -= LEARNING_RATE * W_2_gradient[j][i] / BATCH_SIZE;
        }
    }

    for (int i = 0; i < 256; i++) {
        B_1[i] -= LEARNING_RATE * B_1_gradient[i] / BATCH_SIZE;
        for (int j = 0; j < 784; j++) {
            W_1[j][i] -= LEARNING_RATE * W_1_gradient[j][i] / BATCH_SIZE;
        }
    }
}

void validate() {
    for (int i = 0; i < 1000; i++) {
        for (int j = 0; j < 256; j++) {
            double dot_product = 0;
            for (int k = 0; k < 784; k++) {
                dot_product += X_val[i][k] * W_1[k][j];
            }
            dot_product += B_1[j];

            //apply ReLU to the final dot product for activation
            dot_product = max(dot_product, 0.0);
            layer1_val[i][j] = dot_product;
        }
    }
    
    //layer 2
    for (int i = 0; i < 1000; i++) {
        for (int j = 0; j < 128; j++) {
            double dot_product = 0;
            for (int k = 0; k < 256; k++) {
                dot_product += layer1_val[i][k] * W_2[k][j];
            }
            dot_product += B_2[j];

            //apply ReLU to the final dot product for activation
            dot_product = max(dot_product, 0.0);
            layer2_val[i][j] = dot_product;
        }
    }

    //layer 3
    for (int i = 0; i < 1000; i++) {
        //sum tracking for softmax activation at end;
        double max_val = -DBL_MAX;
        for (int j = 0; j < 10; j++) {
            double dot_product = 0;
            for (int k = 0; k < 128; k++) {
                dot_product += layer2_val[i][k] * W_3[k][j];
            }
            dot_product += B_3[j];

            max_val = max(max_val, dot_product);
            layer3_val[i][j] = dot_product;
        }
        //apply softmax in order to get final predictions
        double sum = 0;
        for (int j = 0; j < 10; j++) {
            layer3_val[i][j] = exp(layer3_val[i][j] - max_val);
            sum += layer3_val[i][j];
        }
        for (int j = 0; j < 10; j++) {
            layer3_val[i][j] /= sum;
        }
    }   
}

void train() {
    int batches = 8000 / BATCH_SIZE;
    int epochs = 50;
    for (int epoch = 0; epoch < epochs; epoch++) {
        /*
        if (epoch != 0 && epoch % 5 == 0) {
            LEARNING_RATE *= 0.5;
        }
        */
        double total_train_error = 0;

        for (int j = 0; j < batches; j++) {
            vector<vector<double>> batch(X_train.begin() + j * BATCH_SIZE, X_train.begin() + j * BATCH_SIZE + BATCH_SIZE);
            vector<int> labels(Y_train.begin() + j * BATCH_SIZE, Y_train.begin() + j * BATCH_SIZE + BATCH_SIZE);
            forward_propagate(batch);
            back_propagate(batch, labels);

            for (int k = 0; k < BATCH_SIZE; k++) {
                total_train_error -= log(layer3[k][labels[k]]);
            }

            if (j == batches - 1) {
                double val_error = 0;
                int correct = 0;
                validate();
                for (int i = 0; i < 1000; i++) {
                    val_error -= log(layer3_val[i][Y_val[i]]);
                    int predicted = max_element(layer3_val[i].begin(), layer3_val[i].end()) - layer3_val[i].begin();
                    if (predicted == Y_val[i]) correct++;
                }

                cout << "Epoch " << epoch + 1 
                    << ", Train Error: " << total_train_error / ((j + 1) * BATCH_SIZE)
                    << ", Validation Error: " << val_error / 1000
                    << ", Validation Accuracy: " << static_cast<double>(correct) / 10 << "%" << endl;
                }
            }
    }
}

int main() {
    //a function to read in training data and initialize weights and biases to random values
    init();

    //the training function
    train();

    return 0;
}