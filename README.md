# SplatPredict

## TODO

### Naive Bayes Classifier

1. [x] Support for continuous training
2. [ ] Discard A1, A2 order information?
   1. [ ] Consider them as weapons for `A` and `B` and multiply probability?
      1. [ ] Treat A1, A2, A3, A4 weapons as the same
      2. [ ] Add combined columns in `model_params` to tell the model to combine the columns when calculating probability and predicting
   2. [ ] But this fail to consider who is holding the weapon
      1. [ ] but the naive bayes assumption assume that each feature is independent
      2. [ ] now we have dependent columns, what should we do?
3. [ ] Current issue
   1. [ ] NB Training accuracy is around 50%, which is not good
   2. [ ] Consider more complex model?
      1. [ ] Ensemble learning algorithm
         1. [ ] Boosting
            1. [ ] Search for more info
      2. [ ] Use Decision tree?

### Web Interface

1. [ ] Write a server to handle
   1. [ ] Use thread to communicate between server request and NBC predication
