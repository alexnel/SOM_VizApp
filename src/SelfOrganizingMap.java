package barebone;

import java.util.*;
import java.io.*;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import java.lang.Math.*;
import javax.swing.JFrame;

class SelfOrganizingMap {
    
    int L;
    int neuronsPerColumn;
    int neuronsPerRow;
    int E; //reduntant, used for convenience
    boolean hexagonalRectangular; //reduntant, used for convenience
    double U[][];
    double neuronPosition[][];
    //double finalPos[][];
    double umat[][];
    double planecomp[][];
    int stepcount;
    GridDisplay out;
    JFrame frame;
    
    //SelfOrganizingMap(){}
    
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
                boolean hexagonalLattice, double standardDeviation, Random randomGenerator, int steps) {                
        
        L = features;
        this.neuronsPerColumn = neuronsPerColumn;
        this.neuronsPerRow = neuronsPerRow;
        E = neuronsPerColumn*neuronsPerRow;
        hexagonalRectangular = hexagonalLattice;
        U = new double[L][E];
        planecomp = new double[neuronsPerRow][neuronsPerColumn];
        umat = new double[2*neuronsPerRow-1][2*neuronsPerColumn-1];
        neuronPosition = new double[E][2];
        stepcount = steps;
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
        
        if(stepcount>0)
        {
//            XYSeriesCollection dataset = new XYSeriesCollection();
//
//            //Boys (Age,weight) series
//            XYSeries series1 = new XYSeries("Neurons");
//
//            for (int i =0; i<E; i++)
//            {
//                series1.add(neuronPosition[i][0], neuronPosition[i][1]);
//            }
//
//            dataset.addSeries(series1);

            display(neuronPosition, "U-Matrix");
            
            //out = new output(dataset);
    //        out.revalidate();
    //        out.repaint();
            //out.setVisible(true);
        }

    
        
    }
    
    /*
    Uses random number generator with a value very likely to be distinct from
    any other invocation of this constructor.
    */
    SelfOrganizingMap(int features, int neuronsPerColumn, int neuronsPerRow,
                boolean hexagonalLattice, double standardDeviation) { 
        this(features, neuronsPerColumn, neuronsPerRow, hexagonalLattice,
                standardDeviation, new Random(), 0);
    }
    
    /*
    Uses random number generator with a value very likely to be distinct from
    any other invocation of this constructor and also a standard deviation equal
    to 0.75.
    */
    SelfOrganizingMap(int features, int neuronsPerColumn, int neuronsPerRow, boolean hexagonalLattice, int stepcount) { 
        this(features, neuronsPerColumn, neuronsPerRow, hexagonalLattice,  0.75, new Random(), stepcount);
    }
    
    /*
    Uses random number generator with a value very likely to be distinct from
    any other invocation of this constructor and also a standard deviation equal
    to 0.75. The grid type is set to be the hexagonal one.
    */
    SelfOrganizingMap(int features, int neuronsPerColumn, int neuronsPerRow) { 
        this(features, neuronsPerColumn, neuronsPerRow, true,  0.75, new Random(), 0);
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
    void learningStep(double x[], double sigma, double learningRate, int stepstop, int steps) {
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
    
        
    if (stepstop>stepcount)
    {
        
//        System.out.println(Integer.toString(stepcount));
//        XYSeriesCollection dataset = new XYSeriesCollection();   
//    
//        XYSeries series1 = new XYSeries("Neurons");
//    
//        for (int i =0; i<U.length; i++)
//        {
//            series1.add(U[i][0], U[i][1]);
//        }
//    
//        dataset.addSeries(series1);
//    
        try        
        {
            Thread.sleep(200);
        } 
        catch(InterruptedException ex) 
        {
            Thread.currentThread().interrupt();
        };
                
        
        makeUmat();
  
//        //out.dispose();
//        out.revalidate();
//        out.repaint();
//        out.update(dataset);
//        out.setVisible(true);
        
       // out.revalidate();
    }//end step count
        stepcount++;    
    
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
    
    void trainWorkbench(double samples[][], int stepstop) {
        //all the following parameters are rough estimates mainly meant for
        //demonstrational purposes
        int epochs = 200;
        int steps = 7000;
        stepcount=0;
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
            learningStep(x, sigma, learningRate, stepstop, steps);
        } 
        
        // - or - //
        
//        //batch training
//        for (int loop = 0; loop <= epochs; loop++) {
//            double sigma = (finalSigma - initialSigma) * loop / epochs + initialSigma;
//            learningEpoch(samples, sigma);
//        } 

        saveParameters("trained SOM model");
    }  
    
    public static void main(String [] args) throws IOException {        
        Scanner scan = new Scanner(System.in);
        
        System.out.print("Enter  dataset file name: ");
        String input = scan.nextLine();
        
        String menu = "";
        
        while (!menu.equalsIgnoreCase("x"))
        {
            System.out.println("Menu");
            System.out.println("C - display Component Plane");
            System.out.println("V - display Component Plane");
            System.out.println("U - display U-Matrix");
            System.out.println("X - exit program");
            System.out.print("Enter in menu item you wish to view: ");
            menu = scan.nextLine();

        
//            System.out.print("Do you wish to view training? (yes/no) ");
//            String watchtrain = scan.nextLine();

            int stepstop = 0;
//            if ("yes".equals(watchtrain))
//            {
//                System.out.print("How many steps of training do you wish to view? ");
//                stepstop = scan.nextInt();
//            }
            

            //transform the dataset in a condensed format based on the floating-point
            //number specification
            IOFiles.fileNormalForm(input,
                    "data in double precision IEEE754");

            int numOfSamples = countLines(input);
            int features = countFeatures(input);


            //load the data in an array, rows: samples, columns: features
            double data[][] = IOFiles.fileToArray("data in double precision IEEE754",
                    numOfSamples, features);
            //adjust the value range of each feature in the [0,1] interval
            data = DataManipulation.adjustPerColumnValueRange(data, true);
            int dimX = 21; //13;
            int dimY = 17; //11;
            SelfOrganizingMap som = new SelfOrganizingMap(features, dimX, dimY, false, stepstop); 
            som.reinitializeCodebookVectors(data);    
            som.trainWorkbench(data, stepstop);
            
            
            if (menu.equalsIgnoreCase("c"))
            {
                System.out.print("Enter the feature number you wish to view: ");
                int feature = scan.nextInt();
                if (feature>features)
                {
                    System.out.print("ERROR: feature out of bounds");
                    return;
                }
                som.compPlane(feature);
            }
            else if (menu.equalsIgnoreCase("v"))
            {
                System.out.print("Enter the number of the input vector you wish to view: ");
                int vector = scan.nextInt();
                if(vector>numOfSamples)
                {
                    System.out.print("ERROR: vector number out of bounds");
                    return;
                }
                som.vecAct(input, vector, features);
            }
            else if (menu.equalsIgnoreCase("u"))
            {
                som.makeUmat();
            }
            else
            {
                return;
            }
            
            
//            if ("yes".equals(watchtrain))
//            {
//                som.makeUmat();
//            }
        
        }//end while menu
        scan.close();
    }
    
    
//    void redisplay(double[][] grid)
//    {
//        double maxcolour = grid[0][0];
//        double mincolour = grid[0][0];
//        
//        for (int row=0; row<grid.length; row++)
//        {
//            for (int col=0; col<grid[0].length; col++)
//            {
//                maxcolour = Math.max(maxcolour, grid[row][col]);
//                mincolour = Math.min(mincolour, grid[row][col]);
//            }
//        }
//        
//        int [][] gridcolours = new int[grid.length][grid[0].length];
//        
//        //there are 10 colours to choose from in the scale, lighter is closer, darker further away
//        double segment = (maxcolour-mincolour)/10; 
//        for (int row=0; row<grid.length; row++)
//        {
//            for (int col=0; col<grid[0].length; col++)
//            {
//                gridcolours[row][col] = 10 - (int) Math.round((grid[row][col]-mincolour)/segment);
//            }
//        }
//        
//        int row = gridcolours.length;
//        int col = gridcolours[0].length;
//  
//
////        out.revalidate();
////        out.repaint();
////        out.setVisible(true);
//            
////        JFrame frame = new JFrame("Game");
//        GridDisplay map = new GridDisplay(gridcolours, row, col);
//        frame.revalidate();
//        frame.repaint();
//        frame.add(map);
////        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
//        frame.pack();
//        frame.setVisible(true);
//    }

    
    void display(double[][] grid, String title)
    {
        double maxcolour = grid[0][0];
        double mincolour = grid[0][0];
        
        for (int row=0; row<grid.length; row++)
        {
            for (int col=0; col<grid[0].length; col++)
            {
                maxcolour = Math.max(maxcolour, grid[row][col]);
                mincolour = Math.min(mincolour, grid[row][col]);
            }
        }
        
        int [][] gridcolours = new int[grid.length][grid[0].length];
        
        //there are 10 colours to choose from in the scale, lighter is closer, darker further away
        double segment = (maxcolour-mincolour)/10; 
        for (int row=0; row<grid.length; row++)
        {
            for (int col=0; col<grid[0].length; col++)
            {
                gridcolours[row][col] = 10 - (int) Math.round((grid[row][col]-mincolour)/segment);
            }
        }
        
        int row = gridcolours.length;
        int col = gridcolours[0].length;
  
        frame = new JFrame(title);
        out = new GridDisplay(gridcolours, row, col);
        frame.add(out);
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.pack();
        frame.setVisible(true);
    }
    
    void compPlane(int feature)
    {
        int count = 0;
        for (int row=0; row<neuronsPerRow; row++)
        {
            for (int col=0; col<neuronsPerColumn; col++)
            {
                planecomp[row][col] = U[feature][count];
                count++;
            }
        }
        display(planecomp, "Component Plane");
    }
    
    
    void vecAct(String filename, int vector, int features) throws IOException
    {
        double[] inputVector = getLine(filename, vector, features);
        double[][] activity = new double[neuronsPerRow][neuronsPerColumn];
        for (int row=0; row<neuronsPerRow; row++)
        {
            for (int col=0; col<neuronsPerColumn; col++)
            {
                activity[row][col] = distanceVect(row,col,inputVector);
            }
        }
        display(activity, "Vector Activity Histogram");
    }
    
    /* calculate the distance between two vectors, input row and col are from umat 
    space and then converted to the U array index */
    double distanceVect(int row1, int col1, double[]vector) {
        double c=0;
        int index1=row1*neuronsPerRow+col1;       
        
        for (int i=0; i<U.length; i++)
        {
            c+= Math.pow(U[i][index1]-vector[i],2);
        }
        
        c = Math.sqrt(c);
        
        return c;
    }
    
    void makeUmat()
    {        
        //find the distances to make up the newly inserted indecies
        for (int row=0; row<umat.length; row++)
        {
            for (int col=0; col<umat[0].length; col++)
            {
                if (row%2!=0)
                {
                    if (col%2==0)
                    {   umat[row][col] = distance(row-1,col,row+1,col);   }
                    else
                    {   umat[row][col] = distance(row-1,col-1,row+1,col+1);   }
                }
                else
                {
                    if (col%2!=0)
                    {   umat[row][col] = distance(row,col-1,row,col+1);   }
                }
            }
        }//end for loop
        
        
        
        //find the new old positions
        for (int row=0; row<umat.length; row++)
        {
            for (int col=0; col<umat[0].length; col++)
            {
                if (row%2==0 && col%2==0)
                {
                    if (row==0)
                    {
                        if (col==0)
                        {   umat[row][col] = ave3(umat[0][1], umat[1][1], umat[1][0]);  }
                        else if (col==umat[0].length-1)
                        {   umat[row][col] = ave3(umat[row][col-1], umat[row+1][col], umat[row+1][col-1]);  }
                        else
                        {   umat[row][col] = ave5(umat[row][col-1], umat[row+1][col-1], umat[row+1][col],umat[row+1][col+1], umat[row][col+1]); }
                    }//first row
                    else if (row==umat.length-1)
                    {
                        if (col==0)
                        {   umat[row][col] = ave3(umat[row-1][col], umat[row-1][col+1], umat[row][col+1]);  }
                        else if (col==umat[0].length-1)
                        {   umat[row][col] = ave3(umat[row][col-1], umat[row-1][col], umat[row-1][col-1]);  }
                        else
                        {   umat[row][col] = ave5(umat[row][col-1], umat[row-1][col-1], umat[row-1][col],umat[row-1][col+1], umat[row][col+1]); }
                    }//last row
                    else
                    {
                        if (col==0)
                        {   umat[row][col] = ave5(umat[row-1][col], umat[row-1][col+1], umat[row][col+1],umat[row+1][col+1], umat[row+1][col]); }
                        else if (col==umat[0].length-1)
                        {   umat[row][col] = ave5(umat[row-1][col], umat[row-1][col-1], umat[row][col-1],umat[row+1][col-1], umat[row+1][col]); }
                        else
                        {   umat[row][col] = ave8(umat[row][col-1], umat[row-1][col-1], umat[row-1][col],umat[row-1][col+1], umat[row][col+1], umat[row+1][col+1],umat[row+1][col], umat[row][col-1]);  }
                    }//middle rows
                }
            }
        }//end for loop
        
        display(umat, "U-matrix");
    }
    
    /* calculate the distance between two vectors, input row and col are from umat 
    space and then converted to the U array index */
    double distance(int row1, int col1, int row2, int col2) {
        double c=0;
        row1=row1/2;
        row2=row2/2;
        col1=col1/2;
        col2=col2/2;
        int index1=row1*neuronsPerRow+col1;
        int index2=row2*neuronsPerRow+col2;        
        
        for (int i=0; i<U.length; i++)
        {
            c+= Math.pow(U[i][index1]-U[i][index2],2);
        }
        
        c = Math.sqrt(c);
        
        return c;
    }
    
    /* find average of surrounding blocks, based on where in the array the block 
    lies a different ave method will be used */
    double ave3(double a, double b, double c)
    {
        return ((a+b+c)/3);
    }
    double ave5(double a, double b, double c, double d, double e)
    {
        return ((a+b+c+d+e)/5);
    }
    double ave8(double a, double b, double c, double d, double e, double f, double g, double h)
    {
        return ((a+b+c+d+e+f+g+h)/8);
    }
    
     public static int countLines(String filename) throws IOException 
    {
        InputStream is = new BufferedInputStream(new FileInputStream(filename));
        try {
            byte[] c = new byte[1024];
            int count = 0;
            int readChars = 0;
            boolean endsWithoutNewLine = false;
            while ((readChars = is.read(c)) != -1) {
                for (int i = 0; i < readChars; ++i) {
                    if (c[i] == '\n')
                        ++count;
                }
                endsWithoutNewLine = (c[readChars - 1] != '\n');
            }
            if(endsWithoutNewLine) {
                ++count;
            } 
            return count;
        } finally {
            is.close();
        }
    }
    
    public static double[] getLine(String filename, int lineNumber, int features) throws IOException 
    {
        double[] vector = new double[features];
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) 
        {
            String line = "";
            for (int i=0; i<lineNumber; i++)
            {
                line=br.readLine();
            }
            String[] splitLine;
            if (line != null) 
            {
                if (line.contains("\t"))
                {
                    splitLine = line.split("\t");
                }
                else if (line.contains(" "))
                {
                    splitLine = line.split(" ");
                }
                else if (line.contains("\r"))
                {
                    splitLine = line.split("\r");
                }
                else
                {
                    splitLine = new String[0];
                }
            }
            else
            {
                splitLine = new String[0];
            }
            
            for(int f=0; f<splitLine.length; f++)
            {
                vector[f] = Double.parseDouble(splitLine[f]);
            }
            
            return vector;
        }
    }
    
    public static int countFeatures(String filename) throws IOException 
    {
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) 
        {
            String line = br.readLine();
            String[] splitLine;
            if (line != null) 
            {
                if (line.contains("\t"))
                {
                    splitLine = line.split("\t");
                }
                else if (line.contains(" "))
                {
                    splitLine = line.split(" ");
                }
                else if (line.contains("\r"))
                {
                    splitLine = line.split("\r");
                }
                else
                {
                    splitLine = new String[0];
                }
            }
            else
            {
                splitLine = new String[0];
            }
            
            int count = splitLine.length;
            return count;
        }
    }
    
}