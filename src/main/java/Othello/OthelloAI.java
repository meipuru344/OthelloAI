package Othello;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.*;


/**
 * Created by miyoshi on 2017/07/05.
 */
public class OthelloAI {

    BufferedReader br;
    PrintWriter pw;

    public OthelloAI() {

        float[] teacheroutput = {
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,1,0,0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,
        };

        float[] teacherinput = {
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,
                0,0,0,1,2,0,0,0,
                0,0,0,2,1,0,0,0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,
                0,0,0,0,0,0,0,0,
        };
        float eta = (float)0.01;
        float sum = 0;

        INDArray teacherInput;
        INDArray teacherOutput;
        INDArray outputLayerWeight = Nd4j.rand(64,100);
        INDArray outputLayerBias   = Nd4j.rand(64,  1);
        INDArray outputWeightDelta = Nd4j.zeros(64,100);
        INDArray outputBiasDelta   = Nd4j.zeros(64,  1);
        INDArray outputOutput      = Nd4j.zeros(64,  1);
        INDArray outputUvector     = Nd4j.zeros(64,  1);

        INDArray hiddenLayerWeight = Nd4j.rand(100,64);
        INDArray hiddenLayerBias   = Nd4j.rand(100, 1);
        INDArray hiddenDelta       = Nd4j.zeros(100, 1);
        INDArray hiddenWeightDelta = Nd4j.zeros(100,64);
        INDArray hiddenBiasDelta   = Nd4j.zeros(100, 1);
        INDArray hiddenOutput      = Nd4j.zeros(100, 1);
        INDArray hiddenUvector     = Nd4j.zeros(100, 1);

        //テキストからデータのインプット
        String str;
        File file = new File("/Users/miyoshi/desktop/nd4j/kihuFixed.txt");


        for(int i=0;i<100;i++){
            try{br = new BufferedReader(new FileReader(file));
            }catch(IOException e){System.out.println("ファイルが見つかりません");}
            int count = 0;
            try{//while((str=br.readLine())!=null) {
                for(int s=0;s<100;s++){str=br.readLine();//棋譜の数指定
                String[] buffer = str.split(" ");
                for (int j = 0; j < 64; j++) {
                    teacherinput[j] = (float) Integer.parseInt(buffer[j]);
                }
                teacherInput = Nd4j.create(teacherinput, new int[]{64, 1});
                int tO = Integer.parseInt(buffer[64]) - 1 + (Integer.parseInt(buffer[65]) - 1) * 8;
                //teacherOutputが0~63であるか確認する
                if(-1 < tO && tO < 64){
                    for(int k=0;k<64;k++){teacheroutput[k]=0;}
                    teacheroutput[tO]=1;
                    teacherOutput = Nd4j.create(teacheroutput,new int[]{64,1});

                    //順方向計算
                    hiddenUvector = hiddenLayerBias.add(hiddenLayerWeight.mmul(teacherInput));
                    hiddenOutput = Transforms.sigmoid(hiddenUvector);
                    outputUvector = outputLayerBias.add(outputLayerWeight.mmul(hiddenOutput));
                    sum = Transforms.exp(outputUvector).sumNumber().floatValue();
                    outputOutput = Transforms.exp(outputUvector).div(sum);

                    //逆伝搬法
                    outputBiasDelta = outputOutput.sub(teacherOutput);
                    outputWeightDelta = outputBiasDelta.mmul(hiddenOutput.transpose());
                    hiddenDelta = outputLayerWeight.transpose().mmul(outputBiasDelta);
                    hiddenBiasDelta = hiddenOutput.mul(hiddenDelta.mul(Nd4j.ones(hiddenOutput.shape()).sub(hiddenOutput)));
                    hiddenWeightDelta = hiddenBiasDelta.mmul(teacherInput.transpose());

                    //学習量実装
                    outputLayerWeight = outputLayerWeight.sub(outputWeightDelta.mul(eta));
                    outputLayerBias = outputLayerBias.sub(outputBiasDelta.mul(eta));
                    hiddenLayerWeight = hiddenLayerWeight.sub(hiddenWeightDelta.mul(eta));
                    hiddenLayerBias = hiddenLayerBias.sub(hiddenBiasDelta.mul(eta));

                    count++;
                    // System.out.println("times:"+count);
                }}}catch(IOException E){}
            System.out.println("epoch:" + (i+1));
        }
        //System.out.println("outputLayerWeight:\n"+outputLayerWeight);
        //System.out.println("outputLayerBias:\n"+outputLayerBias);
        //System.out.println("hiddenLayerWeight:\n"+hiddenLayerWeight);
        //System.out.println("hiddenLayerBias:\n"+hiddenLayerBias);

        //parameter.txtへの出力
        file = new File("/Users/miyoshi/desktop/nd4j/parameter.txt");
        try{pw = new PrintWriter(new BufferedWriter(new FileWriter(file)));}catch(IOException C){}
        printW(pw,outputLayerWeight);
        printB(pw,outputLayerBias.transpose());
        printW(pw,hiddenLayerWeight);
        printB(pw,hiddenLayerBias.transpose());
        pw.close();

    }

    public void printW(PrintWriter p, INDArray array){
        int[] shape = array.shape();
        int row=shape[0], column=shape[1];
        for(int i=0;i<row;i++){
            for(int j=0;j<column;j++){
                p.print(array.getFloat(array.index(i,j))+" ");
            }p.println();
            p.flush();
        }
    }
    public void printB(PrintWriter p,INDArray array){
        int length = array.length();
        for(int i=0;i<length;i++){
            p.print(array.getFloat(i)+" ");
        }p.flush();
    }

    public static void main(String args[]){
        new OthelloAI();
        System.out.println("Hello, This is OthelloAI Ver 2.01");
    }

}
