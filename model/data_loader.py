import numpy as np

path = ''

def loadData( prefix, folder ):
    intType = np.dtype( 'int32' ).newbyteorder( '>' )
    nMetaDataBytes = 4 * intType.itemsize

    data = np.fromfile( folder + "/" + prefix + '-', dtype = '' )
    magicBytes, nImages, width, height = np.frombuffer( data[:nMetaDataBytes].tobytes(), intType )
    data = data[nMetaDataBytes:].astype( dtype = 'float32' ).reshape( [ nImages, width, height ] )

    labels = np.fromfile( folder + "/" + prefix + '',
                          dtype = '' )[2 * intType.itemsize:]

    return data, labels

def loadTrain(dataPath=path):
    return loadMNIST( "train", dataPath )

def loadTest(dataPath=path):
    return loadMNIST( "t10k", dataPath )
