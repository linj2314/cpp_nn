#include "main.hpp"

using namespace std;
using namespace cv;

using m1D = vector<double>;
using m2D = vector<m1D>;
using m3D = vector<m2D>;
using m4D = vector<m3D>;

const int BATCH_SIZE = 50;
const double LEARNING_RATE = 0.01;

//train, validation and testing datasets
vector<int> Y_train(8000, 0);
m3D X_train(8000, m2D(30, m1D(30, 0)));

vector<int> Y_val(2000, 0);
m3D X_val(2000, m2D(30, m1D(30, 0)));

vector<int> Y_test(1000, 0);
m3D X_test(1000, m2D(30, m1D(30, 0)));

//kernels and weights
m4D K_1(32, m3D(3, m2D(3, m1D(1))));
m4D K_2(32, m3D(3, m2D(3, m1D(32))));
m4D K_4(64, m3D(3, m2D(3, m1D(32))));
m4D K_5(64, m3D(3, m2D(3, m1D(64))));
m4D K_7(256, m3D(3, m2D(3, m1D(64))));
m4D K_8(256, m3D(3, m2D(3, m1D(256))));
m2D W_10(256, m1D(1024));
m2D W_11(1024, m1D(10));

//bias terms
m1D B_1(32, 0);
m1D B_2(32, 0);
m1D B_4(64, 0);
m1D B_5(64, 0);
m1D B_7(256, 0);
m1D B_8(256, 0);
m1D B_10(1024, 0);
m1D B_11(10, 0);

//convolution, pooling, and classic neural network layers
m4D L_1(BATCH_SIZE, m3D(28, m2D(28, m1D(32, 0))));
m4D L_2(BATCH_SIZE, m3D(26, m2D(26, m1D(32, 0))));
m4D P_3(BATCH_SIZE, m3D(15, m2D(15, m1D(32, 0))));
m4D L_4(BATCH_SIZE, m3D(13, m2D(13, m1D(64, 0))));
m4D L_5(BATCH_SIZE, m3D(11, m2D(11, m1D(64, 0))));
m4D P_6(BATCH_SIZE, m3D(7, m2D(7, m1D(64, 0))));
m4D L_7(BATCH_SIZE, m3D(5, m2D(5, m1D(256, 0))));
m4D L_8(BATCH_SIZE, m3D(3, m2D(3, m1D(256, 0))));
m4D P_9(BATCH_SIZE, m3D(1, m2D(1, m1D(256, 0))));

m2D L_10(BATCH_SIZE, m1D(1024));
m2D L_11(BATCH_SIZE, m1D(10));

//to store indexes for max pooling layers
vector<vector<vector<vector<int>>>> PI_3(BATCH_SIZE, vector<vector<vector<int>>>(13, vector<vector<int>>(13, vector<int>(32, 0))));
vector<vector<vector<vector<int>>>> PI_6(BATCH_SIZE, vector<vector<vector<int>>>(5, vector<vector<int>>(5, vector<int>(64, 0))));
vector<vector<vector<vector<int>>>> PI_9(BATCH_SIZE, vector<vector<vector<int>>>(1, vector<vector<int>>(1, vector<int>(256, 0))));

//dropout locations
vector<vector<vector<vector<bool>>>> PD_3(BATCH_SIZE, vector<vector<vector<bool>>>(13, vector<vector<bool>>(13, vector<bool>(32, false))));
vector<vector<vector<vector<bool>>>> PD_6(BATCH_SIZE, vector<vector<vector<bool>>>(5, vector<vector<bool>>(5, vector<bool>(64, false))));
vector<vector<vector<vector<bool>>>> PD_9(BATCH_SIZE, vector<vector<vector<bool>>>(1, vector<vector<bool>>(1, vector<bool>(256, false))));
vector<vector<bool>> LD_10(BATCH_SIZE, vector<bool>(1024, false));

//generator for random numbers
unsigned seed = chrono::system_clock::now().time_since_epoch().count();
mt19937 rng(seed);
uniform_real_distribution<> uni_dist(0.0, 1.0);

void print(m4D & m) {
    for (auto & i : m) {
        for (auto & j : i) {
            for (auto & k : j) {
                for (auto & l : k) {
                    cout << l << " ";
                }
            }
        }
    }
}

void gen_kernel(m4D & kernel, uniform_real_distribution<double> & dist){
    for (auto & i : kernel) {
        for (auto & j : i) {
            for (auto & k : j) {
                for (auto & l : k) {
                    l = dist(rng);
                }
            }
        }
    }
};

void init_dropout(vector<vector<vector<vector<bool>>>> & D) {
    for (auto & i : D) {
        for (auto & j : i) {
            for (auto & k : j) {
                for (auto l : k) {
                    if (uni_dist(rng) <= 0.2) {
                        l = true;
                    }
                }
            }
        }
    }
}

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
                    X_train[ind_train][j + 1][k + 1] = (static_cast<double>(image.at<uchar>(j, k)) - 127.5) / 255.0;
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
                    X_val[ind_val][j + 1][k + 1] = (static_cast<int>(image.at<uchar>(j, k)) - 127.5) / 255.0;
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
    m3D X_val_temp(2000, m2D(30, m1D(30)));

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
    m3D X_train_temp(8000, m2D(30, m1D(30)));

    for (int i = 0; i < 8000; i++) {
        Y_train_temp[i] = Y_train[indexes2[i]];
        X_train_temp[i] = X_train[indexes2[i]];
    }

    swap(Y_train, Y_train_temp);
    swap(X_train, X_train_temp);

    uniform_real_distribution<double> dist1(-1.0 * sqrt(6.0/297.0), sqrt(6.0/297));
    uniform_real_distribution<double> dist2(-1.0 * sqrt(6.0/576), sqrt(6.0/576));
    uniform_real_distribution<double> dist3(-1.0 * sqrt(6.0/864), sqrt(6.0/864));
    uniform_real_distribution<double> dist4(-1.0 * sqrt(6.0/1152), sqrt(6.0/1152));
    uniform_real_distribution<double> dist5(-1.0 * sqrt(6.0/2880), sqrt(6.0/2880));
    uniform_real_distribution<double> dist6(-1.0 * sqrt(6.0/4608), sqrt(6.0/4608));
    normal_distribution<double> dist7(0, sqrt(2.0/256.0));
    normal_distribution<double> dist8(0, sqrt(2.0/1034.0));

    gen_kernel(K_1, dist1);
    gen_kernel(K_2, dist2);
    gen_kernel(K_4, dist3);
    gen_kernel(K_5, dist4);
    gen_kernel(K_7, dist5);
    gen_kernel(K_8, dist6);

    for (auto & i : W_10) {
        for (auto & j : i)
            j = dist7(rng);
    }

    for (auto & i : W_11) {
        for (auto & j : i)
            j = dist8(rng);
    }

    init_dropout(PD_3);
    init_dropout(PD_6);
    init_dropout(PD_9);

    for (auto & i : LD_10) {
        for (auto j : i) {
            if (uni_dist(rng) <= 0.5) {
                j = true;
            }
        }
    }
};

void ff_convolution(m4D & input, m4D & output, m4D & kernel, m1D & bias) {
    for (int i = 0; i < BATCH_SIZE; i++) {
        for (int j = 0; j < kernel.size(); j++) {
            for (int k = 0; k < output[0].size(); k++) {
                for (int l = 0; l < output[0][0].size(); l++) { 
                    double dot_prod = 0;
                    for (int ki = 0; ki < 3; ki++) {
                        for (int li = 0; li < 3; li++) {
                            for (int mi = 0; mi < kernel[0][0][0].size(); mi++) {
                                dot_prod += input[i][k + ki][l + li][mi] * kernel[j][ki][li][mi];
                            }
                        }
                    }
                    output[i][k][l][j] = max(dot_prod + bias[j], 0.0);
                }
            }
        }
    }
}

void ff_pooling(m4D & input, m4D & output, vector<vector<vector<vector<int>>>> & indexes, vector<vector<vector<vector<bool>>>> & dropout, int padding) {
    for (int i = 0; i < BATCH_SIZE; i++) {
        for (int j = 0; j < input[0][0][0].size(); j++) {
            for (int k = 0; k < output[0].size() - padding * 2; k++) {
                for (int l = 0; l < output[0][0].size() - padding * 2; l++) {
                    double max_val = input[i][k * 2][l * 2][j];
                    int max_ind = 0;
                    for (int ki = 0; ki < 2; ki++) {
                        for (int li = 0; li < 2; li++) {
                            if (input[i][k * 2 + ki][l * 2 + li][j] > max_val) {
                                max_val = input[i][k * 2 + ki][l * 2 + li][j];
                                max_ind = ki * 2 + li;
                            }
                        }
                    }
                    indexes[i][k][l][j] = max_ind;
                    if (dropout[i][k][l][j]) {
                        output[i][k + padding][l + padding][j] = 0;
                    } else {
                        output[i][k + padding][l + padding][j] = max_val * (1.0 / (1.0 - 0.2));
                    }
                }
            }
        }
    }
}

void feed_forward(m3D & input) {
    m4D input_4D(BATCH_SIZE, m3D(30, m2D(30, m1D(1))));
    for (int i = 0; i < BATCH_SIZE; i++) {
        for (int j = 0; j < 28; j++) {
            for (int k = 0; k < 28; k++) {
                input_4D[i][j + 1][k + 1][0] = input[i][j + 1][k + 1];
            }
        }
    }


    ff_convolution(input_4D, L_1, K_1, B_1);
    ff_convolution(L_1, L_2, K_2, B_2);
    ff_pooling(L_2, P_3, PI_3, PD_3, 1);

    ff_convolution(P_3, L_4, K_4, B_4);
    ff_convolution(L_4, L_5, K_5, B_5);
    ff_pooling(L_5, P_6, PI_6, PD_6, 1);
    
    ff_convolution(P_6, L_7, K_7, B_7);
    ff_convolution(L_7, L_8, K_8, B_8);
    ff_pooling(L_8, P_9, PI_9, PD_9, 0);
    
    //fully connected layers
    //not worth making functions for
    {
        for (int i = 0; i < BATCH_SIZE; i++) {
            for (int j = 0; j < 1024; j++) {
                double dot_prod = 0;
                for (int k = 0; k < 256; k++) {
                    dot_prod += P_9[i][0][0][k] * W_10[k][j];
                }
                L_10[i][j] = max(dot_prod + B_10[j], 0.0);
                if (LD_10[i][j]) {
                    L_10[i][j] = 0;
                } else {
                    L_10[i][j] *= 2; 
                }
            }
        }

        for (int i = 0; i < BATCH_SIZE; i++) {
            for (int j = 0; j < 10; j++) {
                double dot_prod = 0;
                for (int k = 0; k < 1024; k++) {
                    dot_prod += L_10[i][k] * W_11[k][j];
                }
                L_11[i][j] = dot_prod + B_11[j];
            }

            double max_val = *max_element(L_11[i].begin(), L_11[i].end());
            double sum = 0;
            for (int j = 0; j < 10; j++) {
                L_11[i][j] = exp(L_11[i][j] - max_val);
                sum += L_11[i][j];
            }
            for (int j = 0; j < 10; j++) {
                L_11[i][j] /= sum;
            }
        }
    } 
}

m4D flip(m4D & m) {
    int num_filters = m.size();
    int depth = m[0][0][0].size();
    m4D ret(num_filters, m3D(3, m2D(3, m1D(depth, 0))));

    for (int i = 0; i < num_filters; i++) {
        for (int j = 0; j < depth; j++) {
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    ret[i][k][l][j] = m[i][2 - k][2 - l][j];
                }
            }
        }
    }

    return ret;
}

void input_cnn_gradients(m4D & kernel, m4D & output_delta, m4D & input_delta, int padding) {
    m4D K_flipped = flip(kernel);
    for (int i = 0; i < BATCH_SIZE; i++) {
        for (int j = 0; j < kernel.size(); j++) {
            for (int k = 0; k < input_delta[0].size() - padding * 2; k++) {
                for (int l = 0; l < input_delta[0][0].size() - padding * 2; l++) {
                    for (int m = 0; m < kernel[0][0][0].size(); m++) {
                        double dot_prod = 0;
                        for (int ki = 0; ki < 3; ki++) {
                            for (int li = 0; li < 3; li++) {
                                dot_prod += output_delta[i][k + ki][l + li][j] * K_flipped[j][ki][li][m];
                            }
                        }
                        input_delta[i][k + padding][l + padding][m] = dot_prod;
                    }
                }
            }
        }
    }
}

void K_B_cnn_gradients(m4D & delta, m4D & inputs, m4D & K_gradient, m1D & B_gradient, int padding) {
    for (int i = 0; i < BATCH_SIZE; i++) {
        for (int j = 0; j < K_gradient.size(); j++) {
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    for (int m = 0; m < inputs[0][0][0].size(); m++) {
                        double error = 0;
                        for (int ki = 0; ki < delta[0].size() - 2 * padding; ki++) {
                            for (int li = 0; li < delta[0][0].size() - 2 * padding; li++) {
                                error += inputs[i][k + ki][l + li][m] * delta[i][ki + padding][li + padding][j];
                            }
                        }
                        K_gradient[j][k][l][m] += error;
                    }
                }
            }
            for (int k = 0; k < delta[0].size() - 2 * padding; k++) {
                for (int l = 0; l < delta[0][0].size() - 2 * padding; l++) {
                    B_gradient[j] += delta[i][k + padding][l + padding][j];
                }
            }
        }
    }
}

void undo_pooling(vector<vector<vector<vector<int>>>> & PI, m4D & input_delta, m4D & output_delta, vector<vector<vector<vector<bool>>>> & PD) {
    for (int i = 0; i < BATCH_SIZE; i++) {
        for (int j = 0; j < PI[0].size(); j++) {
            for (int k = 0; k < PI[0][0].size(); k++) {
                for (int l = 0; l < PI[0][0][0].size(); l++) {
                    if (PD[i][j][k][l]) continue;
                    int ji = 0;
                    int ki = 0;
                    switch (PI[i][j][k][l]) {
                        case 1:
                            ji = 1;
                            break;
                        case 2:
                            ki = 1;
                            break;
                        case 3:
                            ji = 1;
                            ki = 1;
                            break;
                    }
                    output_delta[i][j * 2 + ji + 2][k * 2 + ki + 2][l] = input_delta[i][j][k][l]; 
                }
            }
        }
    }
}

void apply_bias_gradient(m1D & b_gradient, m1D & b) {
    for (int i = 0; i < b.size(); i++) {
        b[i] -= b_gradient[i] / BATCH_SIZE * LEARNING_RATE;
    }
}

void apply_kernel_gradient(m4D & k_gradient, m4D & kernel) {
    for (int i = 0; i < kernel.size(); i++) {
        for (int j = 0; j < kernel[0].size(); j++) {
            for (int k = 0; k < kernel[0][0].size(); k++) {
                for (int l = 0; l < kernel[0][0][0].size(); l++) {
                    kernel[i][j][k][l] -= k_gradient[i][j][k][l] / BATCH_SIZE * LEARNING_RATE;
                }
            }
        }
    }
}

void apply_weight_gradient(m2D & w_gradient, m2D & w) {
    for (int i = 0; i < w.size(); i++) {
        for (int j = 0; j < w_gradient[0].size(); j++) {
            w[i][j] -= w_gradient[i][j] / BATCH_SIZE * LEARNING_RATE;
        }
    }
}

void back_propagate(vector<int> & labels, m3D & batch) {
    m4D batch_4D(BATCH_SIZE, m3D(30, m2D(30, m1D(1, 0))));
    for (int i = 0; i < BATCH_SIZE; i++) {
        for (int j = 0; j < 28; j++) {
            for (int k = 0; k < 28; k++) {
                batch_4D[i][j + 1][k + 1][0] = batch[i][j + 1][k + 1];
            }
        }
    }

    m2D delta11(BATCH_SIZE, m1D(10, 0));
    m2D delta10(BATCH_SIZE, m1D(1024, 0));
    m4D delta9(BATCH_SIZE, m3D(1, m2D(1, m1D(256, 0))));

    m2D W_11_gradient(1024, m1D(10, 0));
    m2D W_10_gradient(256, m1D(1024, 0));
    
    m1D B_11_gradient(10, 0);
    m1D B_10_gradient(1024, 0);

    //back propagation for fully connected layers
    //relatively short so didn't put into functions
    {
        for (int i = 0; i < BATCH_SIZE; i++) {
            for (int j = 0; j < 10; j++) {
                delta11[i][j] = L_11[i][j] - (j == labels[i] ? 1 : 0);
            }
        }

        for (int i = 0; i < BATCH_SIZE; i++) {
            for (int j = 0; j < 1024; j++) {
                double error = 0;
                for (int k = 0; k < 10; k++) {
                    error += W_11[j][k] * delta11[i][k];
                }
                if (L_10[i][j] > 0 && !LD_10[i][j]) {
                    delta10[i][j] = error;
                }
            }
        }

        for (int i = 0; i < BATCH_SIZE; i++) {
            for (int j = 0; j < 256; j++) {
                double error = 0;
                for (int k = 0; k < 1024; k++) {
                    error += W_10[j][k] * delta10[i][k];
                }
                if (P_9[i][0][0][j] > 0) {
                    delta9[i][0][0][j] = error;
                }
            }
        }

        for (int i = 0; i < BATCH_SIZE; i++) {
            for (int j = 0; j < 10; j++) {
                B_11_gradient[j] += delta11[i][j];
                for (int k = 0; k < 1024; k++) {
                    W_11_gradient[k][j] += L_10[i][k] * delta11[i][j];
                }
            }

            for (int j = 0; j < 1024; j++) {
                B_10_gradient[j] += delta10[i][j];
                for (int k = 0; k < 256; k++) {
                    W_10_gradient[k][j] += P_9[i][0][0][k] * delta10[i][j];
                }
            }
        }
    }

    //back propagation for convolutional layers
    m4D delta8(BATCH_SIZE, m3D(7, m2D(7, m1D(256, 0)))); //2 layers padding; real size 3 x 3
    m4D delta7(BATCH_SIZE, m3D(7, m2D(7, m1D(256, 0)))); //1 layer padding; real size 5 x 5
    m4D delta6(BATCH_SIZE, m3D(5, m2D(5, m1D(64, 0)))); //0 layer padding; real size 5 x 5
    m4D delta5(BATCH_SIZE, m3D(15, m2D(15, m1D(64, 0)))); //2 layers padding; real size 11 x 11
    m4D delta4(BATCH_SIZE, m3D(15, m2D(15, m1D(64, 0)))); //1 layer padding; real size 13 x 13
    m4D delta3(BATCH_SIZE, m3D(13, m2D(13, m1D(32, 0)))); //0 layer padding; real size 13 x 13
    m4D delta2(BATCH_SIZE, m3D(30, m2D(30, m1D(32, 0)))); //2 layer padding; real size 26 x 26
    m4D delta1(BATCH_SIZE, m3D(30, m2D(30, m1D(32, 0)))); //1 layer padding; real size 28 x 28

    m4D K_8_gradient(256, m3D(3, m2D(3, m1D(256, 0))));
    m4D K_7_gradient(256, m3D(3, m2D(3, m1D(64, 0))));
    m4D K_5_gradient(64, m3D(3, m2D(3, m1D(64))));
    m4D K_4_gradient(64, m3D(3, m2D(3, m1D(32))));
    m4D K_2_gradient(32, m3D(3, m2D(3, m1D(32))));
    m4D K_1_gradient(32, m3D(3, m2D(3, m1D(1))));

    m1D B_8_gradient(256, 0);
    m1D B_7_gradient(256, 0);
    m1D B_5_gradient(64, 0);
    m1D B_4_gradient(64, 0);
    m1D B_2_gradient(32, 0);
    m1D B_1_gradient(32, 0);
    
    //back propagate through convolutional blocks
    
    undo_pooling(PI_9, delta9, delta8, PD_9);

    K_B_cnn_gradients(delta8, L_7, K_8_gradient, B_8_gradient, 2);
    input_cnn_gradients(K_8, delta8, delta7, 1);
    
    K_B_cnn_gradients(delta7, P_6, K_7_gradient, B_7_gradient, 1);
    input_cnn_gradients(K_7, delta7, delta6, 0); //seg fault
    
    undo_pooling(PI_6, delta6, delta5, PD_6);
    
    K_B_cnn_gradients(delta5, L_4, K_5_gradient, B_5_gradient, 2);
    input_cnn_gradients(K_5, delta5, delta4, 1);
    
    K_B_cnn_gradients(delta4, P_3, K_4_gradient, B_4_gradient, 1);
    input_cnn_gradients(K_4, delta4, delta3, 0);
    
    undo_pooling(PI_3, delta3, delta2, PD_3);

    K_B_cnn_gradients(delta2, L_1, K_2_gradient, B_2_gradient, 2);
    input_cnn_gradients(K_2, delta2, delta1, 1);
    
    K_B_cnn_gradients(delta1, batch_4D, K_1_gradient, B_1_gradient, 1);

    //apply all gradients
    apply_weight_gradient(W_11_gradient, W_11);
    apply_weight_gradient(W_10_gradient, W_10);

    apply_bias_gradient(B_11_gradient, B_11);
    apply_bias_gradient(B_10_gradient, B_10);
    apply_bias_gradient(B_8_gradient, B_8);
    apply_bias_gradient(B_7_gradient, B_7);
    apply_bias_gradient(B_5_gradient, B_5);
    apply_bias_gradient(B_4_gradient, B_4);
    apply_bias_gradient(B_2_gradient, B_2);
    apply_bias_gradient(B_1_gradient, B_1);

    apply_kernel_gradient(K_8_gradient, K_8);
    apply_kernel_gradient(K_7_gradient, K_7);
    apply_kernel_gradient(K_5_gradient, K_5);
    apply_kernel_gradient(K_4_gradient, K_4);
    apply_kernel_gradient(K_2_gradient, K_2);
    apply_kernel_gradient(K_1_gradient, K_1);
}

void train() {
    int batches = 8000 / BATCH_SIZE;
    //#pragma omp parallel for
    for (int i = 0; i < 50; i++) {
        for (int j = 0; j < batches; j++) {
            cout << "Epoch " << i << " batch " << j << "\n";
            m3D batch = m3D(X_train.begin() + j * BATCH_SIZE, X_train.begin() + j * BATCH_SIZE + BATCH_SIZE);
            vector<int> labels = vector<int>(Y_train.begin() + j * BATCH_SIZE, Y_train.begin() + j * BATCH_SIZE + BATCH_SIZE);
            feed_forward(batch);
            back_propagate(labels, batch);
        }

        //validation set here
        val_accuracy = 0;
        for (int j = 0; j < 1000 / BATCH_SIZE; j++) {
            m3D batch = m3D(X_val.begin() + j * BATCH_SIZE, X_val.begin() + j * BATCH_SIZE + BATCH_SIZE);
            vector<int> labels = vector<int>(Y_val.begin() + j * BATCH_SIZE, Y_val.begin() + j * BATCH_SIZE + BATCH_SIZE);
            feed_forward(batch);
            for (int k = 0; k < BATCH_SIZE; k++) {
                if (max_element(L_11[k].begin(), L_11[k].end()) - L_11[k].begin() == labels[k]) {
                    val_accuracy += 0.1;
                }
            }
        }
        cout << "Validation accuracy: " << val_accuracy << "%\n";
    }
}

int main() {
    //omp_set_num_threads(6);
    
    init();

    train();

    return 0;
}
