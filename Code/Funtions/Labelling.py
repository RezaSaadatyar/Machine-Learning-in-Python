import numpy as np
from sklearn import  preprocessing

# ========================================== Preparing data ==============================================
def labelling(labels, conv_binary="off"):
    """
    This function encodes or decodes labels based on the specified type.
    
    Parameters:
    labels (numpy array): Array of labels to be encoded or decoded. 
                          Can be either a multi-dimensional array (one-hot encoded) or a 1-dimensional array.
    """

    # Check if the labels array is multi-dimensional and the type_labels is binary
    if conv_binary.lower() == "on":
        # Convert to one-hot encoded format
        one_hot_y = np.zeros((labels.size, labels.max() + 1))# Initialize a zero matrix for one-hot encoding
        one_hot_y[np.arange(labels.size), labels] = 1        # Set the appropriate indices to 1
        labels = one_hot_y                                   # Update labels to the one-hot encoded array
    else:
        # Convert one-hot encoded array to original labels
        if len(labels.shape) > 1:                            # Check if the labels array is multi-dimensional
            labels = np.argmax(labels, axis=1)               # Use argmax to find the original labels
        mod = preprocessing.LabelEncoder()                   # Initialize the LabelEncoder
        labels = mod.fit_transform(labels)                   # Fit and transform the labels using the LabelEncoder

    return labels                                            # Return the transformed labels
