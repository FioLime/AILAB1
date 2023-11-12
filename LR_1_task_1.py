import numpy as np
from sklearn import preprocessing

input_labels = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white']

encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)

print("\n Label mapping: ")
for i, item in enumerate(encoder.classes_):
    print(item, '-->', i)

    test_labels = ['green', 'red', 'black']
    encoder_values = encoder.transform(test_labels)
    print("\n Labels =", test_labels)
    print("Encoder values =", list (encoder_values))

    encoded_values = [3, 0, 4, 1]
    decoded_list = encoder.inverse_transform(encoded_values)
    print("\n Encoded values =", encoded_values)
    print("Decoded labels =", list (decoded_list))