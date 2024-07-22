#include "main.hpp"

using namespace std;
using namespace cv;

//an 8000 x {1, 784} data structure to hold our input data
vector<int> Y_train;
vector<vector<int>> X_train;

//weights and biases vectors
vector<vector<double>> W_1(784, vector<double>(256, 0));
vector<vector<double>> W_2(256, vector<double>(128, 0));
vector<vector<double>> W_3(128, vector<double>(10, 0));
vector<double> B_1(256, 0);
vector<double> B_2(128, 0);
vector<double> B_3(10, 0);

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
            vector<int> v(784, 0);
            Mat image = imread("train/" + simpson_characters[i] + "/" + file_name, cv::IMREAD_GRAYSCALE);
            
            for (int j = 0; j < image.rows; j++) {
                for (int k = 0; k < image.cols; k++) {
                    v[j * 28 + k] = static_cast<int>(image.at<uchar>(j, k));
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
    vector<vector<int> *> X_train_temp(8000);

    for (int i = 0; i < 8000; i++) {
        Y_train_temp[indexes[i]] = Y_train[i];
        X_train_temp[indexes[i]] = &X_train[i];
    }

    Y_train = move(Y_train_temp);
    for (int i = 0; i < 8000; i++) {
        X_train[i] = *X_train_temp[i];
    }

    //initializing weights and biases to random values between -0.5 and 0.5
    uniform_real_distribution<> dist(-0.5, 0.5);

    auto rand_gen = [&dist, &rng](){
        return dist(rng);
    };

    for (auto & i : W_1)
        generate(i.begin(), i.end(), rand_gen);
    for (auto & i : W_2)
        generate(i.begin(), i.end(), rand_gen);
    for (auto & i : W_3)
        generate(i.begin(), i.end(), rand_gen);
    generate(B_1.begin(), B_1.end(), rand_gen);
    generate(B_2.begin(), B_2.end(), rand_gen);
    generate(B_3.begin(), B_3.end(), rand_gen);
};

//forwards propagation function; takes a batch as input and returns the errors
vector<double> forward_propagate(vector<vector<int>> & batch, vector<int> labels) {
    return {};
}

//backwards propagation function; nudges weights and biases to minimize error using gradient descent
void back_propagate() {

}

void train() {
    for (int i = 0; i < 80; i++) {
        vector<vector<int>> batch(X_train.begin() + i * 100, X_train.begin() + i * 100 + 100);
        vector<int> labels(Y_train.begin() + i * 100, Y_train.begin() + i * 100 + 100);
        vector<double> errors = forward_propagate(batch, labels);
    }
}

int main() {
    //a function to read in training data and initialize weights and biases to random values
    init();

    //the training function
    train();

    return 0;
}