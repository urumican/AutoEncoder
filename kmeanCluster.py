from sklearn.cluster import KMeans
import operator

def clusterAccurarcy(predictions, n_clusters=10):
    km = KMeans(n_clusters=n_clusters)

    clusters = km.fit_predict(predictions)

    #Count labels per cluster
    labelCount = {}

    for idx in xrange(len(test_y)):
        cluster = clusters[idx]
        label = test_y[idx]

        if cluster not in labelCount:
            labelCount[cluster] = {}

        if label not in labelCount[cluster]:
            labelCount[cluster][label] = 0

        labelCount[cluster][label] += 1

    #Majority Voting
    clusterLabels = {}
    for num in xrange(n_clusters):
        maxLabel = max(labelCount[num].iteritems(), key=operator.itemgetter(1))[0]
        clusterLabels[num] = maxLabel
    #print clusterLabels
    #Number of errors
    errCount = 0
    for idx in xrange(len(test_y)):
        cluster = clusters[idx]
        clusterLabel = clusterLabels[cluster]
        label = test_y[idx]

        if label != clusterLabel:
            errCount += 1

    return errCount/float(len(test_y))

print "PCA Accurarcy: %f%%" % (clusterAccurarcy(pca_test)*100)
print "AE Accurarcy: %f%%" % (clusterAccurarcy(ae_test)*100)
