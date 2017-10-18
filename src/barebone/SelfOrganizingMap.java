package barebone;

import java.util.*;
import java.io.*;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

class SelfOrganizingMap {
    
    int L;
    int neuronsPerColumn;
    int neuronsPerRow;
    int E; //reduntant, used for convenience
    boolean hexagonalRectangular; //reduntant, used for convenience
    double U[][];
    double neuronPosition[][];
    
    /*
    Generic constructor for a SOM network. The dimensionality/number of the inputs
    is defined by the features variable. The lattice consists of the appropriate
    number of per column and per row neurons. For instance a 4x7 SOM is represented
    by an array with 4 rows (i.e. neuronsPerColumn) and 7 columns (i.e neuronsPerRow).
    Positions are stored starting from the lower left neuron and proceeding in a
    left-to-right down-to-up fashion until the last neuron which is stored in the
    upper right. There is an option for either a hexagonal or rectangular output
    neural map. For facilitating the reproducibility of results a Random class
    object is provided and also the standard deviation value of the corresponding
    Gaussian distribution is set.
    */
    SelfOrganizingMap(int features, int neuronsPerColumn, int neuronsPerRow,
                boolean hexagonalLattice, double standardDeviation, Random randomGenerator) {                
        L = features;
        this.neuronsPerColumn = neuronsPerColumn;
        this.neuronsPerRow = neuronsPerRow;
        E = neuronsPerColumn*neuronsPerRow;
        hexagonalRectangular = hexagonalLattice;
        U = new double[L][E];
        neuronPosition = new double[E][2];

        //initialization of the weights
        for (int l = 0; l < L; l++) {
            for (int e = 0; e < E; e++) {
                U[l][e] = randomGenerator.nextGaussian() * standardDeviation;
            }
        }
        
        //hexagonal grid positions, the Euclidean distance between the closest
        //neurons equals one (1), typically the closest neighboring neurons
        //are six (6)
        if (hexagonalRectangular) {
            double stepX = 1.0;
            double offsetX = 0.5;
            double stepY = Math.sin(Math.toRadians(60));
            double posX;
            double posY = 0.0;
            for (int dimY = 0; dimY < neuronsPerColumn; dimY++) {
                if (dimY % 2 == 0) {
                    posX = 0.0;
                }
                else {
                    posX = offsetX;
                }
                for (int dimX = 0; dimX < neuronsPerRow; dimX++) {
                    neuronPosition[dimY * neuronsPerRow+dimX][0] = posX;
                    neuronPosition[dimY * neuronsPerRow+dimX][1] = posY;
                    posX += stepX;
                }
                posY += stepY;
            }
        }
        //rectangular grid positions, the Euclidean distance between the closest
        //neurons equals one (1), typically the closest neighboring neurons
        //are four (4)
        else {
            double step = 1.0;
            double posX;
            double posY = 0.0;
            for (int dimY = 0; dimY < neuronsPerColumn; dimY++) {
                posX = 0.0;
                for (int dimX = 0; dimX < neuronsPerRow; dimX++) {
                    neuronPosition[dimY * neuronsPerRow+dimX][0] = posX;
                    neuronPosition[dimY * neuronsPerRow+dimX][1] = posY;
                    posX += step;
                }
                posY += step;
            }            
        }        
        
        XYSeriesCollection dataset = new XYSeriesCollection();

    //Boys (Age,weight) series
    XYSeries series1 = new XYSeries("Neurons");
    
    for (int i =0; i<E; i++)
    {
        series1.add(neuronPosition[i][0], neuronPosition[i][1]);
    }
    
    dataset.addSeries(series1);
        
        
        output s = new output(dataset);
        s.setVisible(true);
        
    }
    
    /*
    Uses random number generator with a value very likely to be distinct from
    any other invocation of this constructor.
    */
    SelfOrganizingMap(int features, int neuronsPerColumn, int neuronsPerRow,
                boolean hexagonalLattice, double standardDeviation) { 
        this(features, neuronsPerColumn, neuronsPerRow, hexagonalLattice,
                standardDeviation, new Random());
    }
    
    /*
    Uses random number generator with a value very likely to be distinct from
    any other invocation of this constructor and also a standard deviation equal
    to 0.75.
    */
    SelfOrganizingMap(int features, int neuronsPerColumn, int neuronsPerRow, boolean hexagonalLattice) { 
        this(features, neuronsPerColumn, neuronsPerRow, hexagonalLattice,  0.75, new Random());
    }
    
    /*
    Uses random number generator with a value very likely to be distinct from
    any other invocation of this constructor and also a standard deviation equal
    to 0.75. The grid type is set to be the hexagonal one.
    */
    SelfOrganizingMap(int features, int neuronsPerColumn, int neuronsPerRow) { 
        this(features, neuronsPerColumn, neuronsPerRow, true,  0.75, new Random());
    }
    
    /*
    The codebook vectors' elements can be (re)initialized based on the minimum
    and maximum values of the samples' features they are set to represent. Values
    are taken from uniform distributions with appropriate centers and spreads so
    as to cover the whole value range.
    */
    void reinitializeCodebookVectors(double samples[][]) {
        double lowerUpperLimits[][];
        Random variable = new Random();
        
        lowerUpperLimits = DataManipulation.perColumnMinMax(samples);
        for (int l = 0; l < L; l++) {
            for (int e = 0; e < E; e++) {
                U[l][e] = variable.nextDouble() * (lowerUpperLimits[1][l] - lowerUpperLimits[0][l]) + 
                        lowerUpperLimits[0][l];
            }
        }
    }
    
    /*
    All model parameters are saved in the respective (filepath and) file.  The
    employed order is the following: L, neuronsPerColumn, neuronsPerRow, E,
    hexagonalRectangular, U and neuronPosition. All parameters except from 
    hexagonalRectangular are stored in their native byte encoding (4 bytes for
    integers and 8 bytes for doubles).
    */
    void saveParameters(String file) {
        int parameterVector[] = new int[5];
        
        parameterVector[0] = L;
        parameterVector[1] = neuronsPerColumn;
        parameterVector[2] = neuronsPerRow;
        parameterVector[3] = E;
        parameterVector[4] = hexagonalRectangular ? 1 : 2;
        
        IOFiles.arrayToFile(parameterVector, file, false);
        IOFiles.arrayToFile(U, file, true);
        IOFiles.arrayToFile(neuronPosition, file, true);
    }
    
    /*
    All model parameters are retrieved from the respective (filepath and) file.
    The employed order is in analogy to the storing order, namely: L, neuronsPerColumn,
    neuronsPerRow, E, hexagonalRectangular, U and neuronPosition. It should be
    kept in mind that except from hexagonalRectangular all the other parameters
    are stored in their native byte encoding.
    */
    void importParameters(String file) {
        try
        {
            FileInputStream fis = new FileInputStream(file);
            BufferedInputStream bis = new BufferedInputStream(fis);
            DataInputStream dis = new DataInputStream(bis);

            int parameterVector[] = IOFiles.fileStreamToIntArray(dis, 5);
            L = parameterVector[0];
            neuronsPerColumn = parameterVector[1];
            neuronsPerRow = parameterVector[2];
            E = parameterVector[3];     
            hexagonalRectangular = (parameterVector[4] == 1);
            U = IOFiles.fileStreamToArray(dis, L, E);
            neuronPosition = IOFiles.fileStreamToArray(dis, E, 2);            
            dis.close();
        }
        catch(IOException e)
        {
            System.out.println("ERROR: "+e.toString());
        }           
    }

    /*
    Estimation of the Gaussian distances (i.e. neighborhood parameters) between a
    neuron c and the rest of the neurons with respect to the given sigma value.
    On the HEXAGONAL grid the closest neighboring neurons have a squared
    Euclidean distance of 1, the second closest neighboring neurons (diagonal)
    have a squared Euclidean distance of 3, the third closest neighboring neurons
    have a squared Euclidean distance of 4 e.t.c. On the RECTANGULAR grid
    the closest neighboring neurons have a squared Euclidean distance of 1,
    the second closest neighboring neurons (diagonal) have a squared Euclidean
    distance of 2, the third closest neighboring neurons have a squared
    Euclidean distance of 4 e.t.c. As a result for the gaussianDistance between
    the nearest neighbors the following equalities apply:
    spread=0.27  => hce=0.0010502673
    spread=0.33  => hce=0.0101389764
    spread=0.466 => hce=0.1000092880
    spread=0.85  => hce=0.5005531348
    spread=1.32  => hce=0.7505413640
    spread=2.18  => hce=0.9001354750    
    Spread values equal to: euclidean(node[0],node[node.length-1])/sqrt(-8*ln(vL))
    result in gaussian distances between winner and half diameter neighbors
    approximately equal to vL, and also is gaussian distances between winner and
    maximum diameter neighbors approximately equal to vL^4.
    */
    double[] gaussianDistance(int c, double sigma) {
        double sumOfSquaredDifferences;
        double hde[] = new double[E];

        for (int e = 0; e < E; e++) {
            sumOfSquaredDifferences = 0.0;
            for (int dim = 0; dim < neuronPosition[0].length; dim++) {
                sumOfSquaredDifferences += (neuronPosition[c][dim] - neuronPosition[e][dim]) *
                        (neuronPosition[c][dim] - neuronPosition[e][dim]);
            }
            hde[e] = Math.exp(-sumOfSquaredDifferences / (2 * sigma * sigma));
        } 
        return hde;
    }
    
    /*
    With respect to a two-dimensional (or an one-dimensional) grid, the returned
    values result in gaussian distances between winner and half diameter neighbors
    approximately equal to gaussianDistance, and also in gaussian distances
    between winner and maximum diameter neighbors approximately equal to
    gaussianDistance^4.
    */
    double sigmaForHalfDiameterNeighborDistance(double gaussianDistance) {
        double sigma;
        
        sigma = Math.hypot(neuronPosition[0][0] - neuronPosition[E - 1][0],
                neuronPosition[0][1] - neuronPosition[E - 1][1]) /
                Math.sqrt(-8.0 * Math.log(gaussianDistance));
        return sigma;
    }
    
    /*
    Calculation of the transpose of codebook matrix U.
    */
    double[][] UT() {
        double transpose[][] = new double[E][L];
        
        for (int i = 0; i < L; i++)
        {
            for (int j = 0; j < E; j++) {
                transpose[j][i] = U[i][j];
            }
        }        
        
        return transpose;
    }
        
    /*
    The usual classical update rule of the SOM.
    */
    void learningStep(double x[], double sigma, double learningRate) {
        int c = -1;
        double hce[];
        double minimumDistance = Double.POSITIVE_INFINITY;
        double squaredEuclidean;
               
        for (int e = 0; e < E; e++) {
            squaredEuclidean = 0.0;
            for (int l = 0; l < L; l++) {
                squaredEuclidean += (x[l] - U[l][e]) * (x[l] - U[l][e]);
            }
            if (squaredEuclidean <= minimumDistance) {
                c = e;
                minimumDistance = squaredEuclidean;
            }
        }              
        hce = gaussianDistance(c, sigma);
        for (int e = 0; e < E; e++) {
            for (int l = 0; l < L; l++) {
                U[l][e] += learningRate * hce[e] * (x[l] - U[l][e]);  
            }
        }
        
    XYSeriesCollection dataset = new XYSeriesCollection();   
    
    XYSeries series1 = new XYSeries("Neurons");
    
    for (int i =0; i<E; i++)
    {
        series1.add(U[i][0], U[i][1]);
    }
    
    dataset.addSeries(series1);
        
        
        output s = new output(dataset);
        s.setVisible(true);
    }
    
    /*
    The batch learning rule for the SOM.
    */
    void learningEpoch(double samples[][], double sigma) {
        double numeratorU[][] = new double[L][E];
        double denominatorU[] = new double[E];
        double squaredEuclidean;
        double minimumDistance;
        int c;
        double hce[];
        
        for (int e = 0; e < E; e++) {
            for (int l = 0; l < L; l++) {            
                numeratorU[l][e] = 0.0;
            }
            denominatorU[e] = 0.0;
        }        
        for (int x = 0; x < samples.length; x++) {
            minimumDistance = Double.POSITIVE_INFINITY;
            c = -1;
            for (int e = 0; e < E; e++) {
                squaredEuclidean = 0.0;
                for (int l = 0; l < L; l++) {
                    squaredEuclidean += (samples[x][l] - U[l][e]) * (samples[x][l] - U[l][e]);
                }
                if (squaredEuclidean <= minimumDistance) {
                    c = e;
                    minimumDistance = squaredEuclidean;
                }
            }              
            hce = gaussianDistance(c, sigma);
            for (int e = 0; e < E; e++) {
                for (int l = 0; l < L; l++) {
                    numeratorU[l][e] += hce[e] * samples[x][l];
                }
                denominatorU[e] += hce[e];
            }
        }
        for (int e = 0; e < E; e++) {
            for (int l = 0; l < L; l++) {
                U[l][e] = numeratorU[l][e] / denominatorU[e];
            }
        }
    }    

    /*
    Detection of the winner (i.e. best-matching) neuron. In case of equidistant
    neurons the last (according to the storing scheme) is proclaimed winner.
    */
    int bestMatchingNeuron(double x[]) {
        int c = -1;
        double minimumDistance = Double.POSITIVE_INFINITY;
        double squaredEuclidean;
      
        for (int e = 0; e < E; e++) {
            squaredEuclidean = 0.0;
            for (int l = 0; l < L; l++) {
                squaredEuclidean += (x[l] - U[l][e]) * (x[l] - U[l][e]);
            }
            if (squaredEuclidean<=minimumDistance) {
                c = e;
                minimumDistance = squaredEuclidean;
            }
        }
        return c;
    }
    
    /*
    Detection of the winner (i.e. best-matching) neurons for a provided data set.
    In case of equidistant neurons the last (according to the storing scheme)
    is proclaimed winner.
    */
    int[] bestMatchingNeuron(double samples[][]) {
        int c[] = new int[samples.length];
        
        for (int x = 0; x < samples.length; x++) {
            c[x] = bestMatchingNeuron(samples[x]);
        }
        
        return c;
    }
    
    /*
    Detection of the second best matching neuron. This translates to a 2nd best
    winner neuron whose codebook vector is the second closest one. In case of
    equidistant neurons the last (according to the storing scheme) is used.
    */
    int secondMatchingNeuron(double x[]) {
        int c = -1;
        int d = -1;
        double squaredEuclidean;
        double minimumDistance = Double.POSITIVE_INFINITY;
        double secondMinimumDistance = Double.POSITIVE_INFINITY;
                                                                                                   
        for (int e = 0; e < E; e++) {
            squaredEuclidean = 0.0;
            for (int l = 0; l < L; l++) {
                squaredEuclidean += (x[l] - U[l][e]) * (x[l] - U[l][e]);
            }            
            if (squaredEuclidean <= minimumDistance) {
                secondMinimumDistance = minimumDistance;
                d = c;
                minimumDistance = squaredEuclidean;
                c = e;
            }
            else if (squaredEuclidean <= secondMinimumDistance) {
                secondMinimumDistance = squaredEuclidean;
                d = e;
            }
        }
        
        return d;
    }    
    
    /*
    Detection of the second best matching neurons. This translates to 2nd best
    winner neurons whose codebook vectors are the second closest ones. In case of
    equidistant neurons the last (according to the storing scheme) is used.
    */
    int[] secondMatchingNeuron(double samples[][]) {
        int d[] = new int[samples.length];
        
        for (int x = 0; x < samples.length; x++) {
            d[x] = secondMatchingNeuron(samples[x]);
        }
        
        return d;
    }
    
    /*
    Topographic Error yields values between [0, 1]. The samples array is
    arranged by having instances as rows and features as columns.
    TE = 0 all best and second best matching neurons for the samples are distant,
    TE = 1 all best and second best matching neurons for the samples are closest
    neighbors.
    Obviously, the map needs to have at least two neurons. 
    */
    double topographicError(double samples[][]) {
        double te = 0.0;
        int n = samples.length;
        int c[];
        int d[];
        double squaredEuclidean;
        
        c = bestMatchingNeuron(samples);
        d = secondMatchingNeuron(samples);
        for (int i = 0; i < n; i++) {            
            squaredEuclidean = 0.0;
            for (int dim = 0; dim < neuronPosition[0].length; dim++) {
                squaredEuclidean += (neuronPosition[c[i]][dim] - neuronPosition[d[i]][dim]) *
                        (neuronPosition[c[i]][dim] - neuronPosition[d[i]][dim]);
            }         
            //since the closest neurons have a squared distance of 1.0
            if (Math.abs(squaredEuclidean - 1.0) < 0.01) {
                te += 1.0;
            }                        
        }
        te /= (double) n;         
        
        return te;
    }    
    
    void trainWorkbench(double samples[][]) {
        //all the following parameters are rough estimates mainly meant for
        //demonstrational purposes
        int epochs = 200;
        int steps = 7000;
        double initialLearningRate = 0.4;
        double finalLearningRate = 0.04;
        //see the comments of  the gaussianDistance() function
        double initialSigma = sigmaForHalfDiameterNeighborDistance(0.1);
        double finalSigma = 0.33; //closest neurons' neighbor values ~= 0.01                  
        
        //online training
        DataManipulation.shuffle(samples, 1000 * samples.length);        
        for (int loop = 0; loop <= steps; loop++) {
            double x[] = samples[loop % samples.length];
            double sigma = (finalSigma - initialSigma) * loop / steps + initialSigma;
            double learningRate = (finalLearningRate - initialLearningRate) * loop / steps +
                    initialLearningRate;
            learningStep(x, sigma, learningRate);
        } 
        
        // - or - //
        
//        //batch training
//        for (int loop = 0; loop <= epochs; loop++) {
//            double sigma = (finalSigma - initialSigma) * loop / epochs + initialSigma;
//            learningEpoch(samples, sigma);
//        } 

        saveParameters("trained SOM model");
    }  
    
    public static void main(String args[]) {        
        int numOfSamples = 657;
        int features = 200;
        //transform the dataset in a condensed format based on the floating-point
        //number specification
        IOFiles.fileNormalForm("glycosyltransferase genes (data).txt",
                "data in double precision IEEE754");
        //load the data in an array, rows: samples, columns: features
        double data[][] = IOFiles.fileToArray("data in double precision IEEE754",
                numOfSamples, features);
        //adjust the value range of each feature in the [0,1] interval
        data = DataManipulation.adjustPerColumnValueRange(data, true);
                
        int dimX = 13;
        int dimY = 11;
        SelfOrganizingMap som = new SelfOrganizingMap(features, dimX, dimY, false);        
        som.reinitializeCodebookVectors(data);        
        som.trainWorkbench(data);
    }
    
    
    
    /*
    ! THE FOLLOWING PIECES OF CODE ARE OLD AND REQUIRE (EXTENSIVE) REWORKING !
    */
//    /*
//    Returns a matrix containing densities (that is, the number of input samples
//    that are assigned to each neuron) necessary for the DensityMatrix graphic
//    display. The variable returned consists of all the already computed densities
//    stored according to a transposelike logic:
//     Q W E
//      R T Y                       |  F  A  U  R  Q  |
//     U I O  <- Density Matrix ->  |  G  S  I  T  W  |
//      A S D                       |  H  D  O  Y  E  |
//     F G H
//    The employed input data (viz. samples) necessary for the corresponding
//    computation, need to be passed.
//    */
//    int[][] densityMatrix(double data[][]) {
//        int frequency[][] = new int[neuronsPerRow][neuronsPerColumn];
//        int c;
//        
//        for (int i = 0; i<neuronsPerRow; i++)
//            for (int j = 0; j<neuronsPerColumn; j++)
//                frequency[i][j] = 0;
//        for (int x = 0; x<data.length; x++) {
//            c = bestMatchingNeuron(data[x]);
//            frequency[c%neuronsPerRow][c/neuronsPerRow]++;
//        }
//        return frequency;    
//    }   
//    /*
//    Given the index of the column that contains the class information (i.e. 
//    classPointer) and the total number of classes present, the density matrices
//    of each individual class are displayed. It is important to note that the first
//    class should be denoted as 0 (or 0.0) the second as 1 or (1.0) and so forth.
//    */
//    void displayClassDensityMatrices(double data[][], int classPointer,
//                                     int numOfClasses) {
//        int perClassCount[] = new int[numOfClasses];
//        double classData[][];
//        
//        for (int i = 0; i<numOfClasses; i++)
//            perClassCount[i] = 0;
//        for (int i = 0; i<data.length; i++)
//            perClassCount[(int)Math.round(data[i][classPointer])]++;                
//        for (int i = 0; i<numOfClasses; i++) {
//            classData = new double[perClassCount[i]][data[0].length-1];
//            int ii = 0;
//            for (int j = 0; j<data.length; j++)
//                if (data[j][classPointer]==i) {
//                    int jj = 0;
//                    for (int k = 0; k<data[0].length; k++)
//                        if (k!=classPointer)
//                            classData[ii][jj++] = data[j][k];
//                    ii++;                    
//                }
////            DensityMatrix.display(densityMatrix(classData));
//        }   
//    }    
//    /*
//    Returns a matrix containing (squared) Euclidean distances necessary for the
//    DistanceMatrix graphic display. The variable returned consists of all the
//    computed distances stored according to a transpose logic:
//     Q W E
//      R T Y                          |  F  A  U  R  Q  |
//       U I O  <- Distance Matrix ->  |  G  S  I  T  W  |
//      A S D                          |  H  D  O  Y  E  |
//     F G H
//    Between to neurons the stored value is their (squared) Euclidean distance.
//    The value of each specific neuron is the average over all its adjacent
//    (squared) Euclidean distances.
//    Variable squaredEuclidean is used for selecting between squared and regular
//    Euclidean distance norms.
//    */
//    double[][] distanceMatrix(boolean squaredEuclidean) {
//        double sparseDistances[][] = new double[E][E];
//        double euclideanDistance;
//        double codebookDistance;
//        int numOfNeighbors;
//        double meanDistances[] = new double[E];
//        double distances[][] = new double[2*neuronsPerRow-1][2*neuronsPerColumn-1];
//             
//        for(int i = 0; i<sparseDistances.length; i++)
//            for(int j = 0; j<sparseDistances[i].length ;j++)
//                sparseDistances[i][j] = Double.NEGATIVE_INFINITY;
//        for(int i = 0; i<sparseDistances.length; i++) {
//            for(int j = 0; j<sparseDistances[i].length; j++) {
//                euclideanDistance = Math.hypot(neuronPosition[i][0]-neuronPosition[j][0],
//                                               neuronPosition[i][1]-neuronPosition[j][1]);
//                if(sparseDistances[i][j]==Double.NEGATIVE_INFINITY&&
//                   euclideanDistance>0.99&&euclideanDistance<1.01) {
//                    codebookDistance = 0.0;
//                    for (int l = 0; l<L; l++)
//                        codebookDistance += (U[l][i]-U[l][j])*(U[l][i]-U[l][j]);
//                    if (!squaredEuclidean)
//                        codebookDistance = Math.sqrt(codebookDistance);
//                    sparseDistances[i][j] = sparseDistances[j][i] = codebookDistance;
//                }
//            }
//        }
//        for(int i = 0; i<sparseDistances.length; i++) {
//           numOfNeighbors = 0;
//           meanDistances[i] = 0.0;
//           for(int j = 0; j<sparseDistances[0].length; j++) {
//              if(sparseDistances[i][j]!=Double.NEGATIVE_INFINITY) {
//                 numOfNeighbors++;
//                 meanDistances[i] += sparseDistances[i][j];
//              }
//           }
//           meanDistances[i] /= (double)numOfNeighbors;
//        }
//        for(int iter2 = 0;iter2<distances[0].length;iter2++) {
//            for(int iter1 = 0;iter1<distances.length;iter1++) {
//                if(iter1%2==0&&iter2%2==0)        
//                    distances[iter1][iter2] = meanDistances[iter1/2+neuronsPerRow*iter2/2];
//                else if(iter1%2==1&&iter2%2==0)          
//                    distances[iter1][iter2] = sparseDistances[(iter1-1)/2+neuronsPerRow*iter2/2][(iter1+1)/2+neuronsPerRow*iter2/2];  
//                else if(iter1%2==0&&iter2%2==1) 
//                    distances[iter1][iter2] = sparseDistances[iter1/2+neuronsPerRow*(iter2-1)/2][iter1/2+neuronsPerRow*(iter2+1)/2];   
//                else if(iter1%2==1&&iter2%2==1) {
//                    if(iter2%4==1) 
//                        distances[iter1][iter2] = sparseDistances[(iter1+1)/2+neuronsPerRow*(iter2-1)/2][(iter1-1)/2+neuronsPerRow*(iter2+1)/2];                        
//                    else if(iter2%4==3)
//                        distances[iter1][iter2] = sparseDistances[(iter1-1)/2+neuronsPerRow*(iter2-1)/2][(iter1+1)/2+neuronsPerRow*(iter2+1)/2];  
//                }
//            }
//        }
//        return distances;
//    }
    /*
    ! IF THINGS WORK WELL ALL THE VISUALIZATION CODES MUST BE EXTENSIVELY REVISED !
    */
    
    
    
}
