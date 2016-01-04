# Get a dependency arch, and add all the possible features to the dict
def add_features_to_dicts(self, dependency_arch):
    self.add_feature_for_dependency((dependency_arch[0], dependency_arch[1]), self.feature_1_dict)
    self.add_feature_for_dependency((dependency_arch[0], ''), self.feature_2_dict)
    self.add_feature_for_dependency((dependency_arch[1], ''), self.feature_3_dict)
    self.add_feature_for_dependency((dependency_arch[2], dependency_arch[3]), self.feature_4_dict)
    self.add_feature_for_dependency((dependency_arch[2], ''), self.feature_5_dict)
    self.add_feature_for_dependency((dependency_arch[3], ''), self.feature_6_dict)
    self.add_feature_for_dependency((dependency_arch[1], (dependency_arch[2], dependency_arch[3])),
                                        self.feature_8_dict)
    self.add_feature_for_dependency(((dependency_arch[0], dependency_arch[1]), dependency_arch[3]),
                                        self.feature_10_dict)
    self.add_feature_for_dependency((dependency_arch[1], dependency_arch[3]), self.feature_13_dict)


# Return the features number that return 1 for a given arch
def get_features_for_arch(features, dependency_arch):
    num_features = []
    if ((dependency_arch[0], dependency_arch[1]), 1) in features:
        num_features.append(features[((dependency_arch[0], dependency_arch[1]), 1)])
    if ((dependency_arch[0], ''), 2) in features:
        num_features.append(features[((dependency_arch[0], ''), 2)])
    if ((dependency_arch[1], ''), 3) in features:
        num_features.append(features[((dependency_arch[1], ''), 3)])
    if ((dependency_arch[2], dependency_arch[3]), 4) in features:
        num_features.append(features[((dependency_arch[2], dependency_arch[3]), 4)])
    if ((dependency_arch[2], ''), 5) in features:
        num_features.append(features[((dependency_arch[2], ''), 5)])
    if ((dependency_arch[3], ''), 6) in features:
        num_features.append(features[((dependency_arch[3], ''), 6)])
    if ((dependency_arch[1], (dependency_arch[2], dependency_arch[3])), 8) in features:
        num_features.append(features[((dependency_arch[1], (dependency_arch[2], dependency_arch[3])), 8)])
    if (((dependency_arch[0], dependency_arch[1]), dependency_arch[3]), 10) in features:
        num_features.append(features[(((dependency_arch[0], dependency_arch[1]), dependency_arch[3]), 10)])
    if ((dependency_arch[1], dependency_arch[3]), 13) in features:
        num_features.append(features[((dependency_arch[1], dependency_arch[3]), 13)])
    return num_features


# Get only frequent features (configurable limits)
def get_frequent_features(self):
    self.add_frequent_feature(self.FEATURE_1_LIMIT, self.feature_1_dict, 1)
    self.add_frequent_feature(self.FEATURE_2_LIMIT, self.feature_2_dict, 2)
    self.add_frequent_feature(self.FEATURE_3_LIMIT, self.feature_3_dict, 3)
    self.add_frequent_feature(self.FEATURE_4_LIMIT, self.feature_4_dict, 4)
    self.add_frequent_feature(self.FEATURE_5_LIMIT, self.feature_5_dict, 5)
    self.add_frequent_feature(self.FEATURE_6_LIMIT, self.feature_6_dict, 6)
    self.add_frequent_feature(self.FEATURE_8_LIMIT, self.feature_8_dict, 8)
    self.add_frequent_feature(self.FEATURE_10_LIMIT, self.feature_10_dict, 10)
    self.add_frequent_feature(self.FEATURE_13_LIMIT, self.feature_13_dict, 13)
