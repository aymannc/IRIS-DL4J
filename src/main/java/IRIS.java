import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

public class IRIS {
    static int batchSize = 16;
    static int outputSize = 3;
    static int classIndex = 4;
    static double learningRate = 0.01;
    static int inputSize = 4;
    static int numHiddenNodes = 10;
    static int nEpochsMax = 100;

    static String[] labels = {"Iris setosa", "Iris versicolor", "Iris virginica"};
    private final int mode;

    private MultiLayerNetwork model;

    public IRIS(int mode) {
        this.mode = mode;
        try {
            if (mode != 0) {
                System.out.println("Loading The model");
                this.model = ModelSerializer.restoreMultiLayerNetwork(new File("model.zip"));
            } else {
                this.buildModel();
            }
        } catch (IOException ignored) {
            this.buildModel();
        }

    }

    public MultiLayerNetwork getModel() {
        return model;
    }

    public static void main(String[] args) throws IOException, InterruptedException {

        IRIS iris = new IRIS(0);

        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();
        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
        InMemoryStatsStorage inMemoryStatsStorage = new InMemoryStatsStorage();
        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(inMemoryStatsStorage);
        //Then add the StatsListener to collect this information from the network, as it trains
        iris.getModel().setListeners(new StatsListener(inMemoryStatsStorage));

        iris.trainModelAndEvaluate();

    }

    public void buildModel() {
        System.out.println("Building the model...");
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(98)
//                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(learningRate))
                .list()
                .layer(0,
                        new DenseLayer.Builder()
                                .nIn(inputSize).nOut(numHiddenNodes)
                                .activation(Activation.SIGMOID).build())
                .layer(1,
                        new OutputLayer.Builder()
                                .nIn(numHiddenNodes).nOut(outputSize)
                                .lossFunction(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
                                .activation(Activation.SOFTMAX).build())
                .build();
        this.model = new MultiLayerNetwork(configuration);
        this.model.init();
    }

    public DataSetIterator getDataSetIterator(String filepath) throws IOException, InterruptedException {
        File testFile = new ClassPathResource(filepath).getFile();
        RecordReader testRecordReader = new CSVRecordReader();
        testRecordReader.initialize(new FileSplit(testFile));
        return new RecordReaderDataSetIterator(testRecordReader, batchSize,
                classIndex, outputSize);
    }

    public void trainModelAndEvaluate() throws IOException, InterruptedException {
        DataSetIterator testSetIterator = this.getDataSetIterator("iris-test.csv");
        DataSetIterator trainDataSetIterator = this.getDataSetIterator("iris-train.csv");

        Evaluation evaluation;
        double startingAccuracy = 0;
        if (this.mode != 0) {
            evaluation = model.evaluate(testSetIterator);
            startingAccuracy = evaluation.accuracy();
            System.out.println("Starting accuracy " + startingAccuracy);
        }
        int i = 1;
        while (startingAccuracy < 1.0) {
            testSetIterator.reset();
            trainDataSetIterator.reset();


            System.out.println("Epoch " + (i++));
            model.fit(trainDataSetIterator);
            evaluation = model.evaluate(testSetIterator);
            System.out.println("old accuracy " + startingAccuracy);
            System.out.println("new accuracy " + evaluation.accuracy());
            if (evaluation.accuracy() > startingAccuracy) {
                startingAccuracy = evaluation.accuracy();
                System.out.println("Saving model !");
                ModelSerializer.writeModel(model, new File("model.zip"), true);
            }

            System.out.println(evaluation.stats());
        }
    }
}
