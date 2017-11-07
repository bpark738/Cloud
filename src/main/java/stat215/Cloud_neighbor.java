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
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.eval.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.graph.ComputationGraph;

import java.io.File;

/**
 * Created by Briton on 11/2/17.
 */
public class Cloud_neighbor {
    private static File baseDir = new File("src/main/resources/");

    public static void main(String[] args) throws Exception{

        int seed = 123;
        double learningRate = 0.008;
        int batchSize = 32;

        final String dirTrain  = "/neighborTrain/";
        final String dirTest  = "/neighborTest/";
        final String crossValSet = "1";

        RecordReader rrTrain1 = new CSVRecordReader(1);
        rrTrain1.initialize(new FileSplit(new File(baseDir + dirTrain + "/n1/" + crossValSet + ".csv")));

        RecordReader rrTrain2 = new CSVRecordReader(1);
        rrTrain2.initialize(new FileSplit(new File(baseDir + dirTrain + "/n2/" + crossValSet + ".csv")));

        RecordReader rrTrain3 = new CSVRecordReader(1);
        rrTrain3.initialize(new FileSplit(new File(baseDir + dirTrain + "/n3/" + crossValSet + ".csv")));

        RecordReader rrTrain4 = new CSVRecordReader(1);
        rrTrain4.initialize(new FileSplit(new File(baseDir + dirTrain + "/n4/" + crossValSet + ".csv")));

        RecordReader rrTrain5 = new CSVRecordReader(1);
        rrTrain5.initialize(new FileSplit(new File(baseDir + dirTrain + "/n5/" + crossValSet + ".csv")));

        MultiDataSetIterator trainIter = new RecordReaderMultiDataSetIterator.Builder(batchSize)
                .addReader("rr1",rrTrain1)
                .addReader("rr2",rrTrain2)
                .addReader("rr3",rrTrain3)
                .addReader("rr4",rrTrain4)
                .addReader("rr5",rrTrain5)
                .addInput("rr1", 1, 8)
                .addInput("rr2", 0, 7)
                .addInput("rr3", 0, 7)
                .addInput("rr4", 0, 7)
                .addInput("rr5", 0, 7)
                .addOutputOneHot("rr1", 0, 2)
                .build();

        RecordReader rrTest1 = new CSVRecordReader(1);
        rrTest1.initialize(new FileSplit(new File(baseDir + dirTest + "/n1/" + crossValSet + ".csv")));

        RecordReader rrTest2 = new CSVRecordReader(1);
        rrTest2.initialize(new FileSplit(new File(baseDir + dirTest + "/n2/" + crossValSet + ".csv")));

        RecordReader rrTest3 = new CSVRecordReader(1);
        rrTest3.initialize(new FileSplit(new File(baseDir + dirTest + "/n3/" + crossValSet + ".csv")));

        RecordReader rrTest4 = new CSVRecordReader(1);
        rrTest4.initialize(new FileSplit(new File(baseDir + dirTest + "/n4/" + crossValSet + ".csv")));

        RecordReader rrTest5 = new CSVRecordReader(1);
        rrTest5.initialize(new FileSplit(new File(baseDir + dirTest + "/n5/" + crossValSet + ".csv")));

        MultiDataSetIterator testIter = new RecordReaderMultiDataSetIterator.Builder(batchSize)
                .addReader("rr1",rrTest1)
                .addReader("rr2",rrTest2)
                .addReader("rr3",rrTest3)
                .addReader("rr4",rrTest4)
                .addReader("rr5",rrTest5)
                .addInput("rr1", 1, 8)
                .addInput("rr2", 0, 7)
                .addInput("rr3", 0, 7)
                .addInput("rr4", 0, 7)
                .addInput("rr5", 0, 7)
                .addOutputOneHot("rr1", 0, 2)
                .build();

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .learningRate(learningRate)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.ADAM)
                .graphBuilder()
                .addInputs("input1", "input2", "input3", "input4", "input5")
                .addLayer("L1", new DenseLayer.Builder()
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .nIn(8).nOut(50)
                        .build(), "input1")
                .addLayer("L2", new DenseLayer.Builder()
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .nIn(8).nOut(50)
                        .build(), "input2")
                .addLayer("L3", new DenseLayer.Builder()
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .nIn(8).nOut(50)
                        .build(), "input3")
                .addLayer("L4", new DenseLayer.Builder()
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .nIn(8).nOut(50)
                        .build(), "input4")
                .addLayer("L5", new DenseLayer.Builder()
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .nIn(8).nOut(50)
                        .build(), "input5")
                .addVertex("merge", new MergeVertex(), "L1", "L2", "L3", "L4", "L5")
                .addLayer("L6", new DenseLayer.Builder()
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .nIn(250).nOut(125).build(), "merge")
                .addLayer("out", new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .nIn(125)
                        .nOut(2).build(), "L6")
                .setOutputs("out")
                .pretrain(false).backprop(true)
                .build();

        ComputationGraph model = new ComputationGraph(conf);

        for ( int n = 0; n < 10; n++) {
            System.out.println("Epoch number: " + n );
            model.fit( trainIter );
        }

        System.out.println("Evaluate model....");

        Evaluation eval = new Evaluation(2);
        while(testIter.hasNext()){
            MultiDataSet t = testIter.next();
            INDArray[] features = t.getFeatures();
            INDArray[] labels = t.getLabels();
            INDArray[] predicted = model.output(features);
            eval.eval(labels[0], predicted[0]);
        }
        testIter.reset();
        System.out.println(eval.stats());

        ROC roc = new ROC();
        while (testIter.hasNext()) {
            MultiDataSet batch = testIter.next();
            INDArray[] output = model.output(batch.getFeatures());
            roc.eval(batch.getLabels(0), output[0]);
        }
        System.out.println("FINAL TEST AUC: " + roc.calculateAUC());
    }
}