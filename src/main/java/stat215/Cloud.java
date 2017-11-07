package stat215;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.deeplearning4j.eval.ROC;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.eval.Evaluation;
import org.nd4j.linalg.dataset.DataSet;

import java.io.File;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.nn.conf.layers.RBM;

/**
 * Created by Briton on 11/2/17.
 */
public class Cloud {
    private static File baseDir = new File("src/main/resources/");

    public static void main(String[] args) throws Exception{

        int seed = 123;
        double learningRate = 0.008;
        int batchSize = 32;
        int nEpochs = 10;

        int numInputs = 8;
        int numOutputs = 2;
        int numHiddenNodes = 50;

        final String crossValSet = "1";
        final String filenameTrain  = "/train/" + crossValSet + ".csv";
        final String filenameTest  = "/test/"  + crossValSet + ".csv";

        RecordReader rrTrain = new CSVRecordReader(1);
        rrTrain.initialize(new FileSplit(new File(baseDir + filenameTrain)));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rrTrain, batchSize,0,2);

        RecordReader rrTest = new CSVRecordReader(1);
        rrTest.initialize(new FileSplit(new File(baseDir + filenameTest)));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize,0,2);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .updater(Updater.ADAM)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunction.MCXENT)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .pretrain(true).backprop(true).build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);


        for ( int n = 0; n < nEpochs; n++) {
            System.out.println("Epoch number: " + n );
            model.fit( trainIter );
        }

        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(numOutputs);
        while(testIter.hasNext()){
            DataSet t = testIter.next();
            INDArray features = t.getFeatureMatrix();
            INDArray labels = t.getLabels();
            INDArray predicted = model.output(features,false);
            eval.eval(labels, predicted);
        }
        testIter.reset();

        System.out.println(eval.stats());

        ROC roc = new ROC();
        while (testIter.hasNext()) {
            DataSet batch = testIter.next();
            INDArray output = model.output(batch.getFeatures());
            roc.eval(batch.getLabels(), output);
        }
        testIter.reset();
        System.out.println("FINAL TEST AUC: " + roc.calculateAUC());
    }
}