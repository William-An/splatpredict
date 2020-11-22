##############
# Name:     Weili An
# email:    an107@purdue.edu
# Date:     Nov. 20th, 2020

import numpy as np  # Required version 1.19.2
import pandas as pd # Required version 1.1.3

# TODO Add model class
class NBC:
    """
    Naive Bayes Classifier
    conditional: {
        "varname": {
            label_class
            y0: ...
            y1: {
                feature condition probability on y1
                total_size: 
                feature_class_count:
                name: count
                name2: count
            }
        }
    }
    prior: {
        total_size:
        y1: count,
        y2: count
    }
    """

    def __init__(self, model_params={}):
        self.accuracy = 0
        self.label_rectifier = lambda x: x
        self.priors = {}
        self.conditional = {}
        self.categories = {}
        self.label_name = ""
        self.feature_names = []
        self.model_params = model_params

    def config(self, model_params):
        # Config parameters for model
        self.model_params = model_params

    def train(self, train_dataset=None, label_name="", train_data=None, train_label=None):
        """
        Note: label name and feature name of the dataset cannot be `total_size` or `feature_class_count` as it is reserved
        train_dataset: training dataset combining data and label
        train_data: training data in pandas dataframe
        train_label: training label in pandas dataframe
        """

        # 0. Create dataset
        try:
            self.label_name = label_name
            self.feature_names = train_dataset.columns.to_list()
            self.feature_names.remove(label_name)
        except AttributeError:
            # No dataset provided, try to combined train_data and train_label
            self.feature_names = train_data.columns
            self.label_name = train_label.columns[0]
            train_dataset = pd.concat([train_data, train_label], axis=1)

        # TODO Create customized class structure to store priors and conditional

        # 1. Initialize conditional dictionary if first time running train()
        feature_names = self.feature_names
        # If self.conditional is an empty dict
        isFirst = not bool(self.conditional)
        if isFirst:
            self.conditional = dict([(feature_name, {}) for feature_name in feature_names])

        # 2. Initialize prior dictionary
        self.label_name = label_name
        # If priors dict is empty
        if isFirst:
            self.priors = dict.fromkeys(train_dataset[self.label_name].unique(), 0)
            self.priors["total_size"] = 0

        # 3. Partition continuous variable if first time running train()
        if isFirst:
            partition = self.model_params["partition"]
            n = train_dataset.shape[0]
            bins_count = int(np.ceil(n**0.5))
            sample_data = train_dataset.sample(frac=self.model_params["sample_frac"])
            for feature_name in feature_names:
                unique_count = sample_data[feature_name].unique().size
                # Partition feature if it is larger than threshold
                if unique_count > self.model_params["max_discretize_count"] and self.model_params["continuous"].get(feature_name, False) == True:
                    if partition == "sqrt":
                        train_dataset[feature_name] = pd.cut(train_dataset[feature_name], bins_count)
                        self.bins_count = bins_count
                    elif partition == "qcut":
                        train_dataset[feature_name] = pd.qcut(train_dataset[feature_name], q=self.model_params["q"], duplicates="drop")
                    else:
                        raise ValueError("Invalid partition scheme for continuous variable")

                    # Get categories to store in order to identify
                    # which category the new value in predict is
                    self.categories[feature_name] = train_dataset[feature_name].cat.categories

        # 4. Calculate conditional and prior
        # Prior
        self.calculate_prior(train_dataset)

        # conditional
        self.calculate_conditional(train_dataset, smoothing=self.model_params["smoothing"])

    def calculate_conditional(self, train_dataset, smoothing=True):
        label_name = self.label_name
        label_values = list(self.priors.keys())
        label_values.remove("total_size")
        for feature in self.feature_names:
            # Calculate the probability conditioning on label value for each feature value
            # using value_counts()
            feature_label_pair = train_dataset[[feature, label_name]]
            feature_values = None
            category_tmp = self.categories.get(feature, None)
            if smoothing and category_tmp is None:
                # Do not smooth interval feature as it will be handled by min and max
                feature_values = self.model_params["smoothing_params"][feature]
            elif category_tmp is not None:
                feature_values = category_tmp   # For sqrt
            else:
                feature_values = feature_label_pair[feature].unique()   # All possible value for this feature

            # Get the feature_conditional dict for the current feature
            feature_conditional = self.conditional.get(feature)
            # If the dict is empty, initialize a new one
            if not bool(feature_conditional):
                feature_conditional = {label: dict() for label in label_values}
            for label_value in label_values:
                # Get feature-single_label
                # pair such that (X, y=yi)
                feature_single_label = feature_label_pair[feature_label_pair[label_name] == label_value]
                feature_single_label = feature_single_label.reset_index(drop=True)

                # Count count(y = yi)
                N = feature_single_label.shape[0]

                # Count feature unique value
                if self.model_params["partition"] == "sqrt":
                    k = self.bins_count
                else:
                    k = len(feature_values)

                # Use the following conditional possibility formula
                # P(X = xi | y = yj) = count(xi and yj)/count(yj)
                # If smoothing,
                # P(X = xi | y = yj) = (count(xi and yj) + 1)/(count(yj) + count(feature_value))

                # Store the count

                # Count (X = xi and y = yj) and store in dict
                total_size = feature_single_label.shape[0]  # total size of the feature-label pair
                feature_count = feature_single_label[feature].value_counts().to_dict()

                # Calculate freq
                # for val in feature_values:
                #     if smoothing:
                #         feature_count[val] = (feature_count.get(val, 0) + 1) / (N + k)
                #     else:
                #         feature_count[val] = feature_count.get(val, 0) / N

                # todo label get same value update?
                for feature_value in feature_values:
                    feature_conditional[label_value][feature_value] = feature_conditional[label_value].get(feature_value, 0) + feature_count.get(feature_value, 0)
                feature_conditional[label_value]["total_size"] = feature_conditional[label_value].get("total_size", 0) + total_size
                if feature_conditional[label_value].get("feature_class_count", None) is None:
                    feature_conditional[label_value]["feature_class_count"] = k
            self.conditional[feature] = feature_conditional

    def calculate_prior(self, train_dataset):
        label_count = train_dataset[self.label_name].value_counts().to_dict()
        total_size = 0
        for label in label_count:
            tmp_count = label_count[label]
            self.priors[label] += tmp_count
            total_size += tmp_count
        self.priors["total_size"] += total_size

    def evaluate(self, test_dataset, label_name):
        test_data = test_dataset.drop(label_name, axis=1)
        test_label = test_dataset[[label_name]]

        size = test_data.shape[0]
        loss_zero_one = 0
        loss_squared = 0
        test_label = self.label_rectifier(test_label)

        for i in range(size):
            predicted_label, predicted_p = self.predict(test_data.iloc[i].to_dict())

            # Calculate 0-1 loss function
            label = test_label[self.label_name].iloc[i]
            if predicted_label != label:
                loss_zero_one += 1  # Count incorrect predicted label
                # Calculate squared loss function
                loss_squared += predicted_p ** 2
            else:
                # Calculate squared loss function
                loss_squared += (1 - predicted_p)**2

        correct = size - loss_zero_one
        loss_zero_one /= size
        loss_squared /= size
        self.accuracy = correct / size

        print(f"ZERO-ONE LOSS={loss_zero_one:.4f}")
        print(f"SQUARED LOSS={loss_squared:.4f} Test Accuracy={self.accuracy:.4f}")
        return loss_zero_one, loss_squared, self.accuracy

    def predict(self, data):
        """
        :param data: single record in dict format
        :return: predict_label: predicted label
        """
        # TODO Perform smoothing here
        predicted_prob = self.priors.copy()
        predicted_prob.pop("total_size")
        for label in predicted_prob.keys():
            predicted_prob[label] /= self.priors["total_size"]
        for label in predicted_prob:
            for feature in data:
                category = self.categories.get(feature, None)
                value = data[feature]
                value_cat_map = value   # Default to be the value, need to classify if it is categorical
                probability_dict = self.conditional[feature][label]

                total_size = probability_dict["total_size"]
                if category is not None:
                    # Need to calculate the category of the data feature value
                    # TODO Handle if the value is already a category value
                    value_cat_map = None
                    try: 
                        tmp = category.contains(value)  # If value 
                        intervalList = category[tmp]
                        if intervalList.size == 1:
                            # Can find a category for the given value
                            value_cat_map = intervalList[0]
                        else:
                            # TODO For real category data, need to handle order issue
                            # TODO But for interval, order is fine
                            # Check min and max
                            minInterval = category[0]
                            if (value < minInterval.left):
                                # If less than min
                                value_cat_map = minInterval
                            else:
                                # Else assume to be larger than max
                                value_cat_map = category[category.size - 1]
                    except NotImplementedError:
                        value_cat_map = value
                
                # Compute prob
                if self.model_params["smoothing"]:
                    tmp_prob = (probability_dict[value_cat_map] + 1) / (total_size + probability_dict["feature_class_count"]) 
                else:
                    tmp_prob = (probability_dict[value_cat_map] / total_size) 
                predicted_prob[label] = predicted_prob[label] * tmp_prob

        # Calculate actual probability
        sum = 0
        for label in predicted_prob:
            sum += predicted_prob[label]
        for label in predicted_prob:
            predicted_prob[label] /= sum

        # Argmax
        max_p = -1
        predicted_label = 0
        for label in predicted_prob:
            if predicted_prob[label] > max_p:
                max_p = predicted_prob[label]
                predicted_label = label
        return predicted_label, max_p

if __name__ == "__main__":
    import sys
    from glob import glob
    # Read data as pandas dataframe
    train_data = pd.read_csv(glob(sys.argv[1])[0], engine="python")
    train_label = pd.read_csv(glob(sys.argv[2])[0], engine="python")
    train_dataset = pd.concat([train_data, train_label], axis=1)
    label_name = train_label.columns[0]

    # Fill NA values in train dataset
    for column in train_data.columns:
        majority_vote = train_dataset[column].value_counts().idxmax()
        train_dataset[column] = train_dataset[column].fillna(majority_vote)
    train_dataset = train_dataset.astype("int32")

    test_data = pd.read_csv(glob(sys.argv[3])[0], engine="python")
    test_label = pd.read_csv(glob(sys.argv[4])[0], engine="python")
    test_dataset = pd.concat([test_data, test_label], axis=1)

    # Drop NA values in test dataset
    for column in test_data.columns:
        test_dataset = test_dataset.dropna()
    test_dataset = test_dataset.astype("int32")

    # Smoothing params: record number of classes for each attributes
    """
    smoothing_params: dict of feature_name:list mapping, list contains all the possible 
        values for this given feature
    continuous: dict of feature_name:Boolean mapping, True if the feature is continous
        and will apply partition to it to discretize the data
    max_discretize_count: if a feature contains feature counts over this value,
        will classify data into ceil(sqrt(n)) bins where n is the number of entries (deprecated)
    sample_frac: percent of data to be considered during counting unique feature value
    partition: method to partition continuous data, can use either "sqrt" or "qcut"
    q: bypass parameter into pd.qcut(), see it for for information
    smoothing: laplace smoothing on the data to prevent zero probability
    """
    model_params = {
        "smoothing_params": {

        },
        "continuous": {

        },
        "max_discretize_count": 10, 
        "sample_frac": 0.5, 
        "partition": "qcut",
        "q":10,
        "smoothing": True
    }