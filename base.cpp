#include <stdio.h>

class SVM {
    public:
        SVM(float *x1, float *x2, int *y, int train_size, int test_size) {
            this->train_size = train_size;
            this->test_size = test_size;

            this->h_x1_train = x1;
            this->h_x2_train = x2;
            this->h_x1_test = x1+train_size;
            this->h_x2_test = x2+train_size;
            this->h_y_train = y;
            this->h_y_test = y+train_size;

            this->trained = false;
            this->made_predictions = false;
        }

        ~SVM() {
            delete[] alpha;
        }

        bool train() {
            trained = true;
            made_predictions = false;


            float lr = .005;

            float w[2] = {0, 0};
            h_intercept = 0;

            for (int i = 0; i < 100; i++) {
                for (int i = 0; i < train_size; i++) {
                    int ypred = w[0] * h_x1_train[i] + w[1] * h_x2_train[i] + h_intercept;
                    int y = h_y_train[i];
                    if (ypred * y <= 0 ) { // Correct prediction
                        // Incorrect prediction
                        w[0] = w[0] + h_x1_train[i] * y * lr;
                        w[1] = w[1] + h_x2_train[i] * y * lr;
                        h_intercept = h_intercept + y * lr;
                    }
                    printf ("w[0] %f\t w[1] %f\t i %f\n", w[0], w[1], h_intercept);
                }
            }

            return true;
        }

        bool make_predictions() {
            if (!trained) {
                printf ("error: not trained\n");
                return false;
            }

            for (int i = 0; i < test_size; i++) {
                if ( (h_x1_test[i] * h_w0 + h_x2_test[i] * h_w1 + h_intercept) > 0) {
                    h_pred[i] = 1;
                } else {
                    h_pred[i] = -1;
                }
            }

            for (int i = 0; i < test_size; i++) {
                printf ("%f\t %f\t %i\t %i\n", h_x1_test[i], h_x2_test[i], h_pred[i], h_y_test[i]);
            }

            made_predictions = true;
            return true;
        }

        // FIXME
        bool calculate_error() {
            if (!trained) {
                printf ("error: not trained\n");
                return false;
            }
            if (!made_predictions) {
                printf ("error: haven't made predictions\n");
                return false;
            }

            return true;
        }

    private:
        int test_size, train_size;
        float *h_x1_test, *h_x2_test, *h_x1_train, *h_x2_train; // Host pointers to independent variables, data stored in main cpp file
        int *h_y_test, *h_y_train; // Host pointers to dependent variables, data stored in main cpp file
        int *h_pred; // Predictions
        float h_w0, h_w1, h_intercept; // Weights and intercept
        bool trained;
        bool made_predictions;

        float *alpha;
};