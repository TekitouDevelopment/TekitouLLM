package com.mc1510ty.TekitouLLM;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ModelCheckpoint {

    // --- 1. 保存メソッド ---
    public static void saveModel(String filePath,
                                 int numLayers, int vectorSize, int num_heads, int dHidden, int vocabSize,
                                 Map<String, Integer> tokenToId,
                                 double[][] embeddingTable,
                                 double[][][][] wq, double[][][][] wk, double[][][][] wv,
                                 double[][][] wGate, double[][][] wUp, double[][][] wDown,
                                 double[][] wOut,
                                 double[][] rmsWeightAttention, double[][] rmsWeightFfn, List<String> merges) throws IOException {

        try (DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(filePath)))) {

            // 1. ハイパーパラメータの書き込み
            dos.writeInt(numLayers);
            dos.writeInt(vectorSize);
            dos.writeInt(num_heads);
            dos.writeInt(dHidden);
            dos.writeInt(vocabSize);

            dos.writeInt(merges.size());
            for (String merge : merges) {
                dos.writeUTF(merge);
            }

            // 2. トークナイザーの辞書（tokenToId）の書き込み
            dos.writeInt(tokenToId.size());
            for (Map.Entry<String, Integer> entry : tokenToId.entrySet()) {
                dos.writeUTF(entry.getKey());
                dos.writeInt(entry.getValue());
            }

            // 3. 埋め込みテーブルの書き込み (double[][])
            write2DArray(dos, embeddingTable);

            // 4. Q, K, V 重みの書き込み (double[][][][])
            write4DArray(dos, wq);
            write4DArray(dos, wk);
            write4DArray(dos, wv);

            // 5. FFN 重みの書き込み (double[][][])
            write3DArray(dos, wGate);
            write3DArray(dos, wUp);
            write3DArray(dos, wDown);

            // 6. 出力層 重みの書き込み (double[][])
            write2DArray(dos, wOut);

            // 7. RMSNorm 重みの書き込み (double[][])
            write2DArray(dos, rmsWeightAttention);
            write2DArray(dos, rmsWeightFfn);

            System.out.println("モデルの保存が完了しました: " + filePath);
        }
    }

    // --- 2. 復元メソッド（読み込み用コンテナを返すクラス等に詰め替えるとスッキリします） ---
    // ※今回は説明用にロードしたデータを格納するためのシンプルなホルダーを想定しています
    public static class LoadedModelData {
        public int numLayers, vectorSize, num_heads, dHidden, vocabSize;
        public Map<String, Integer> tokenToId = new HashMap<>();
        public Map<Integer, String> idToToken = new HashMap<>();
        public double[][] embeddingTable;
        public double[][][][] wq, wk, wv;
        public double[][][] wGate, wUp, wDown;
        public double[][] wOut;
        public double[][] rmsWeightAttention, rmsWeightFfn;
        public List<String> merges = new ArrayList<>();
    }


    public static LoadedModelData loadModel(String filePath) throws IOException {
        LoadedModelData data = new LoadedModelData();

        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(filePath)))) {

            // 1. ハイパーパラメータの読み込み
            data.numLayers = dis.readInt();
            data.vectorSize = dis.readInt();
            data.num_heads = dis.readInt();
            data.dHidden = dis.readInt();
            data.vocabSize = dis.readInt();

            int mergesSize = dis.readInt();
            for (int i = 0; i < mergesSize; i++) {
                data.merges.add(dis.readUTF());
            }

            // 2. トークナイザーの辞書の読み込み
            int vocabSizeMap = dis.readInt();
            for (int i = 0; i < vocabSizeMap; i++) {
                String token = dis.readUTF();
                int id = dis.readInt();
                data.tokenToId.put(token, id);
                data.idToToken.put(id, token);
            }

            // 3. 各重みの読み込み
            data.embeddingTable = read2DArray(dis);
            data.wq = read4DArray(dis);
            data.wk = read4DArray(dis);
            data.wv = read4DArray(dis);
            data.wGate = read3DArray(dis);
            data.wUp = read3DArray(dis);
            data.wDown = read3DArray(dis);
            data.wOut = read2DArray(dis);
            data.rmsWeightAttention = read2DArray(dis);
            data.rmsWeightFfn = read2DArray(dis);

            System.out.println("モデルの読み込みが完了しました: " + filePath);
        }
        return data;
    }

    // --- 配列を読み書きするためのヘルパーメソッド ---
    private static void write2DArray(DataOutputStream dos, double[][] arr) throws IOException {
        dos.writeInt(arr.length);
        dos.writeInt(arr[0].length);
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[i].length; j++) {
                dos.writeDouble(arr[i][j]);
            }
        }
    }

    private static double[][] read2DArray(DataInputStream dis) throws IOException {
        int d1 = dis.readInt();
        int d2 = dis.readInt();
        double[][] arr = new double[d1][d2];
        for (int i = 0; i < d1; i++) {
            for (int j = 0; j < d2; j++) {
                arr[i][j] = dis.readDouble();
            }
        }
        return arr;
    }

    private static void write3DArray(DataOutputStream dos, double[][][] arr) throws IOException {
        dos.writeInt(arr.length);
        dos.writeInt(arr[0].length);
        dos.writeInt(arr[0][0].length);
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[i].length; j++) {
                for (int k = 0; k < arr[i][j].length; k++) {
                    dos.writeDouble(arr[i][j][k]);
                }
            }
        }
    }

    private static double[][][] read3DArray(DataInputStream dis) throws IOException {
        int d1 = dis.readInt();
        int d2 = dis.readInt();
        int d3 = dis.readInt();
        double[][][] arr = new double[d1][d2][d3];
        for (int i = 0; i < d1; i++) {
            for (int j = 0; j < d2; j++) {
                for (int k = 0; k < d3; k++) {
                    arr[i][j][k] = dis.readDouble();
                }
            }
        }
        return arr;
    }

    private static void write4DArray(DataOutputStream dos, double[][][][] arr) throws IOException {
        dos.writeInt(arr.length);
        dos.writeInt(arr[0].length);
        dos.writeInt(arr[0][0].length);
        dos.writeInt(arr[0][0][0].length);
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[i].length; j++) {
                for (int k = 0; k < arr[i][j].length; k++) {
                    for (int m = 0; m < arr[i][j][k].length; m++) {
                        dos.writeDouble(arr[i][j][k][m]);
                    }
                }
            }
        }
    }

    private static double[][][][] read4DArray(DataInputStream dis) throws IOException {
        int d1 = dis.readInt();
        int d2 = dis.readInt();
        int d3 = dis.readInt();
        int d4 = dis.readInt();
        double[][][][] arr = new double[d1][d2][d3][d4];
        for (int i = 0; i < d1; i++) {
            for (int j = 0; j < d2; j++) {
                for (int k = 0; k < d3; k++) {
                    for (int m = 0; m < d4; m++) {
                        arr[i][j][k][m] = dis.readDouble();
                    }
                }
            }
        }
        return arr;
    }
}