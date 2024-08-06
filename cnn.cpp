#include "main.hpp"

using namespace std;
using namespace cv;

using m3D = vector<vector<vector<float>>>;
using m4D = vector<vector<vector<vector<float>>>>;

const int BATCH_SIZE = 1;

//train, validation and testing datasets
vector<int> Y_train(8000, 0);
m3D X_train(8000, vector<vector<float>>(28, vector<float>(28, 0)));

vector<int> Y_val(2000, 0);
m3D X_val(2000, vector<vector<float>>(28, vector<float>(28, 0)));

vector<int> Y_test(1000, 0);
m3D X_test(1000, vector<vector<float>>(28, vector<float>(28, 0)));

//kernels
m4D K_1(32, m3D(3, vector<vector<float>>(3, vector<float>(1))));
m4D K_2(32, m3D(3, vector<vector<float>>(3, vector<float>(32))));
m4D K_3(64, m3D(3, vector<vector<float>>(3, vector<float>(64))));
m4D K_4(64, m3D(3, vector<vector<float>>(3, vector<float>(64))));
m4D K_5(256, m3D(3, vector<vector<float>>(3, vector<float>(256))));
m4D K_6(256, m3D(3, vector<vector<float>>(3, vector<float>(256))));

//bias terms
vector<float> B_1(32, 0);
vector<float> B_2(32, 0);
vector<float> B_3(64, 0);
vector<float> B_4(64, 0);
vector<float> B_5(256, 0);
vector<float> B_6(256, 0);
vector<float> B_7(1024, 0);
vector<float> B_8(10, 0);

//convolution and classic neural network layers
m4D L_1(64, m3D(26, vector<vector<float>>(26, vector<float>(32))));
m4D L_2(64, m3D(24, vector<vector<float>>(24, vector<float>(32))));
m4D L_3(64, m3D(22, vector<vector<float>>(22, vector<float>(64))));
m4D L_4(64, m3D(20, vector<vector<float>>(20, vector<float>(64))));
m4D L_5(64, m3D(18, vector<vector<float>>(18, vector<float>(256))));
m4D L_6(64, m3D(16, vector<vector<float>>(16, vector<float>(256))));

vector<vector<float>> L_7(64, vector<float>(1024));
vector<vector<float>> L_8(64, vector<float>(10));

//generator for random numbers
unsigned seed = chrono::system_clock::now().time_since_epoch().count();
mt19937 rng(seed);
normal_distribution<float> dist(0.0, 1.0);
struct rand_gen{
    double operator()() {
        return dist(rng);
    }
};

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

    int ind_train = 0, ind_val = 0;
    for (int i = 0; i < 10; i++) {
        string cmd = "cd train && cd " + simpson_characters[i] + " && ls > ../../temp.txt";
        auto s = system(cmd.c_str());

        ifstream in = ifstream("temp.txt");
        string file_name;
        while (in >> file_name) {
            Mat image = imread("train/" + simpson_characters[i] + "/" + file_name, cv::IMREAD_GRAYSCALE);
            
            for (int j = 0; j < image.rows; j++) {
                for (int k = 0; k < image.cols; k++) {
                    X_train[ind_train][j][k] = static_cast<float>(image.at<uchar>(j, k)) / 255.0;
                }
            }
            Y_train[ind_train] = i;
            ind_train++;
        }
    }

    for (int i = 0; i < 10; i++) {
        string cmd = "cd test && cd " + simpson_characters[i] + " && ls > ../../temp.txt";
        auto s = system(cmd.c_str());

        ifstream in = ifstream("temp.txt");
        string file_name;
        int ind = 0;
        while (in >> file_name) {
            Mat image = imread("test/" + simpson_characters[i] + "/" + file_name, cv::IMREAD_GRAYSCALE);
            
            for (int j = 0; j < image.rows; j++) {
                for (int k = 0; k < image.cols; k++) {
                    X_val[ind_val][j][k] = static_cast<int>(image.at<uchar>(j, k)) / 255.0f;
                }
            }
            Y_val[ind_val] = i;
            ind_val++;
        }
    }

    //shuffle all data
    vector<int> indexes(2000);
    iota(indexes.begin(), indexes.end(), 0);
    shuffle(indexes.begin(), indexes.end(), rng);

    vector<int> Y_val_temp(2000);
    m3D X_val_temp(2000, vector<vector<float>>(28, vector<float>(28)));

    for (int i = 0; i < 2000; i++) {
        Y_val_temp[i] = Y_val[indexes[i]];
        X_val_temp[i] = X_val[indexes[i]];
    }

    swap(Y_val, Y_val_temp);
    swap(X_val, X_val_temp);

    X_test = m3D(X_val.begin() + 1000, X_val.end());
    Y_test = vector<int>(Y_val.begin() + 1000, Y_val.end());

    Y_val.resize(1000);
    X_val.resize(1000);

    vector<int> indexes2(8000);
    iota(indexes2.begin(), indexes2.end(), 0);
    shuffle(indexes2.begin(), indexes2.end(), rng);

    vector<int> Y_train_temp(8000);
    m3D X_train_temp(8000, vector<vector<float>>(28, vector<float>(28)));

    for (int i = 0; i < 8000; i++) {
        Y_train_temp[i] = Y_train[indexes2[i]];
        X_train_temp[i] = X_train[indexes2[i]];
    }

    swap(Y_train, Y_train_temp);
    swap(X_train, X_train_temp);

    rand_gen rd;
    auto gen_kernel = [&rd](m4D & kernel){
        for (auto & i : kernel) {
            for (auto & j : i) {
                for (auto & k : j) {
                    generate(k.begin(), k.end(), rd);
                }
            }
        }
    };
    gen_kernel(K_1);
    gen_kernel(K_2);
    gen_kernel(K_3);
    gen_kernel(K_4);
    gen_kernel(K_5);
    gen_kernel(K_6);
};

void feed_forward(m3D & input) {
    for (int i = 0; i < input.size(); i++) {
        for (int j = 0; j < K_1.size(); j++) {
            for (int k = 0; k < 26; k++) {
                for (int l = 0; l < 26; l++) { 
                    double dot_prod = 0;
                    for (int ki = 0; ki < 3; ki++) {
                        for (int li = 0; li < 3; li++) {
                            dot_prod += input[i][k + ki][l + li] * K_1[j][ki][li][0];
                        }
                    }
                    L_1[i][k][l][j] = dot_prod;
                }
            }
        }
    }
}

void back_propagate() {

}

void train() {
    int batches = 8000 / BATCH_SIZE;
    for (int i = 0; i < 1; i++) {
        m3D batch = m3D(X_train.begin() + i * BATCH_SIZE, X_train.begin() + i * BATCH_SIZE + BATCH_SIZE);
        feed_forward(batch);
    }
}

int main() {
    init();

    train();

    return 0;
}
