package com.mc1510ty.TekitouLLM;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.spi.AbstractResourceBundleProvider;

public class Main {

    static void main() {

        Scanner scanner = new Scanner(System.in);

        IO.println("=======================================================");
        IO.println(" 適当LLM (C) 2026 TekitouDevelopment, 2023-2026 1510ty");
        IO.println("=======================================================");

        long seed = 12345L;
        Random random = new Random(seed);


        boolean larned = false;
        ModelCheckpoint.LoadedModelData modeldata = null;

        IO.println();


        String omomiPath;

        while (true) {

            IO.print("重みファイルのパスを指定してください (新規学習の場合はそのまま改行): ");
            omomiPath = scanner.nextLine();

            if (omomiPath != null) {
                omomiPath = omomiPath.replace("\"", "").trim();
            }

            if (omomiPath == null || Objects.equals(omomiPath, "")) {
                IO.println("新規学習を開始します");
                break;
            } else if (Files.exists(Path.of(omomiPath))) {
                try {
                    modeldata = ModelCheckpoint.loadModel(omomiPath);
                    IO.println("モデルの読み込みに成功しました");
                    larned = true;
                    break;
                } catch (IOException e) {
                    IO.println("モデルの読み込みに失敗しました");
                    e.printStackTrace();
                }
            } else {
                IO.println("ファイルが見つかりません (新規学習の場合はそのまま改行)");
            }
        }


        // 学習設定
        int numLayers = larned ? modeldata.numLayers : 6; // 層(レイヤー)の数
        int vectorSize = larned ? modeldata.vectorSize : 128; // 次元の数
        int maxSeqLen = 384; // 最大トークン数


        double learningRate = 0.0005; // 学習率
        int batchSize = 12; // 並列数
        int targetVocabSize = 100000; //BPEの目標コンテキスト数

        // 設定や初期化のイメージ
        int num_heads = larned ? modeldata.num_heads : 8; // ヘッド数
        int head_size = vectorSize / num_heads;


        // AdamWのハイパーパラメータ
        double beta1 = 0.9;
        double beta2 = 0.999;
        double eps = 1e-8;
        double weightDecay = 0.01;
        int adamStep = 0;

        // 2. BPEトークナイザーの初期化と学習
        SimpleTokenizer tokenizer = new SimpleTokenizer();
        int vocabSize;

        // ★ allDataset を復活させ、新規学習のときだけ訓練するようにしました
        String[] allDataset = null;

        Path modelSavePath = null;

        // 変数の宣言
        String[] pretrainDataset = null;
        String[] chatDataset = null;

        String omomilearnusedatamoto;

        double stopLoss = 0;

        if (larned) {
            // ロード成功時は保存されていた辞書を復元
            tokenizer.tokenToId = modeldata.tokenToId;
            tokenizer.idToToken = modeldata.idToToken;
            tokenizer.merges = modeldata.merges;
            vocabSize = modeldata.vocabSize;
            omomilearnusedatamoto = modeldata.omomilearnusedatamoto;
        } else {


            List<String> pretrainList;

            while (true) {
                IO.print("事前学習ファイルのパス: ");
                String pathstr = scanner.nextLine();

                if (pathstr != null) {
                    pathstr = pathstr.replace("\"", "").trim();
                }

                Path path = Path.of(pathstr);

                if (pathstr == null || Objects.equals(pathstr, "")) {
                    IO.println("パスが空白またはnullです");
                } else if (!Files.exists(path)) {
                    IO.println("ファイルが見つかりません");
                } else {
                    try {
                        pretrainList = Files.readAllLines(path, StandardCharsets.UTF_8);
                        pretrainDataset = pretrainList.toArray(new String[0]);
                        break;
                    } catch (IOException e) {
                        IO.println("ファイルの読み込みに失敗しました");
                    }
                }

            }

            List<String> chatList;

            while (true) {
                IO.print("対話学習ファイルのパス: ");
                String pathstr = scanner.nextLine();

                if (pathstr != null) {
                    pathstr = pathstr.replace("\"", "").trim();
                }

                Path path = Path.of(pathstr);

                if (pathstr == null || Objects.equals(pathstr, "")) {
                    IO.println("パスが空白またはnullです");
                } else if (!Files.exists(path)) {
                    IO.println("ファイルが見つかりません");
                } else {
                    try {
                        chatList = Files.readAllLines(path, StandardCharsets.UTF_8);
                        chatDataset = chatList.toArray(new String[0]);
                        break;
                    } catch (IOException e) {
                        IO.println("ファイルの読み込みに失敗しました");
                    }
                }

            }

            while (true) {
                IO.print("学習データ一元覧ファイルのパス: ");
                String pathstr = scanner.nextLine();

                if (pathstr != null) {
                    pathstr = pathstr.replace("\"", "").trim();
                }

                Path path = Path.of(pathstr);

                if (pathstr == null || Objects.equals(pathstr, "")) {
                    IO.println("パスが空白またはnullです");
                } else if (!Files.exists(path)) {
                    IO.println("ファイルが見つかりません");
                } else {
                    try {
                        omomilearnusedatamoto = Files.readString(path, StandardCharsets.UTF_8).trim();
                        break;
                    } catch (IOException e) {
                        IO.println("ファイルの読み込みに失敗しました");
                    }
                }

            }


            while (true) {
                IO.print("学習完了時のモデル保存先パス (保存しない場合は空白): ");
                String pathstr = scanner.nextLine();

                if (pathstr != null) {
                    pathstr = pathstr.replace("\"", "").trim();
                }

                Path path = Path.of(pathstr);

                if (pathstr == null || Objects.equals(pathstr, "")) {
                    IO.print("モデルを保存しないでよろしいですか？(y/n): ");
                    String modelsavasinaideiika = scanner.nextLine();
                    if (Objects.equals(modelsavasinaideiika, "y")) {
                        IO.println("モデルを保存しません");
                        modelSavePath = null;
                        break;
                    }
                } else {
                    modelSavePath = path;
                    break;
                }
            }

            while (true) {
                IO.print("学習を停止する誤差率: ");
                String str = scanner.nextLine();

                try {
                    // 文字列を小数に変換する
                    stopLoss = Double.parseDouble(str);
                } catch (NumberFormatException e) {
                    IO.println("doubleでお願いします");
                    continue; // もう一度入力をやり直す
                }

                IO.print(stopLoss + " でよろしいですか？(y/n): ");
                String answer = scanner.nextLine();
                if (Objects.equals(answer, "y")) {
                    break;
                }

            }


            // ロード失敗時はこれまで通りデータセットを結合して訓練
            allDataset = new String[pretrainDataset.length + chatDataset.length];
            System.arraycopy(pretrainDataset, 0, allDataset, 0, pretrainDataset.length);
            System.arraycopy(chatDataset, 0, allDataset, pretrainDataset.length, chatDataset.length);

            tokenizer.train(allDataset, targetVocabSize, 2);

            vocabSize = tokenizer.getVocabSize();


        }

        int dModel = vectorSize;
        int dHidden = larned ? modeldata.dHidden : dModel * 2; // 隠れ層は2倍
        int vocabSizeLocal = vocabSize;

        // 3. 埋め込みテーブルと重みの定義
        double[][] embeddingTable = larned ? modeldata.embeddingTable : new double[vocabSize][vectorSize];
        double[][][][] wq = larned ? modeldata.wq : new double[numLayers][num_heads][vectorSize][head_size];
        double[][][][] wk = larned ? modeldata.wk : new double[numLayers][num_heads][vectorSize][head_size];
        double[][][][] wv = larned ? modeldata.wv : new double[numLayers][num_heads][vectorSize][head_size];
        double[][][] wGate = larned ? modeldata.wGate : new double[numLayers][vectorSize][dHidden];
        double[][][] wUp = larned ? modeldata.wUp : new double[numLayers][vectorSize][dHidden];
        double[][][] wDown = larned ? modeldata.wDown : new double[numLayers][dHidden][vectorSize];
        double[][] wOut = larned ? modeldata.wOut : new double[vectorSize][vocabSizeLocal];

        double[][] rmsWeightAttention = larned ? modeldata.rmsWeightAttention : new double[numLayers][vectorSize];
        double[][] rmsWeightFfn = larned ? modeldata.rmsWeightFfn : new double[numLayers][vectorSize];

        // 新規学習のときだけランダム初期化を実行する
        if (!larned) {
            for (int i = 0; i < vocabSize; i++) {
                for (int j = 0; j < vectorSize; j++) {
                    embeddingTable[i][j] = (random.nextDouble() - 0.5);
                }
            }

            for (int l = 0; l < numLayers; l++) {
                for (int h = 0; h < num_heads; h++) {
                    for (int i = 0; i < vectorSize; i++) {
                        for (int j = 0; j < head_size; j++) {
                            wq[l][h][i][j] = (random.nextDouble() - 0.5) * 0.1;
                            wk[l][h][i][j] = (random.nextDouble() - 0.5) * 0.1;
                            wv[l][h][i][j] = (random.nextDouble() - 0.5) * 0.1;
                        }
                    }
                }
                for (int i = 0; i < vectorSize; i++) {
                    for (int j = 0; j < dHidden; j++) {
                        wGate[l][i][j] = (random.nextDouble() - 0.5) * 0.1;
                        wUp[l][i][j] = (random.nextDouble() - 0.5) * 0.1;
                    }
                }
                for (int i = 0; i < dHidden; i++) {
                    for (int j = 0; j < vectorSize; j++) {
                        wDown[l][i][j] = (random.nextDouble() - 0.5) * 0.1;
                    }
                }
            }
            for (int i = 0; i < vectorSize; i++) {
                for (int j = 0; j < vocabSizeLocal; j++) {
                    wOut[i][j] = (random.nextDouble() - 0.5) * 0.1;
                }
            }

            for (int l = 0; l < numLayers; l++) {
                for (int i = 0; i < vectorSize; i++) {
                    rmsWeightAttention[l][i] = 1.0;
                    rmsWeightFfn[l][i] = 1.0;
                }
            }
        }


        double[] invFreq = new double[vectorSize];
        for (int j = 0; j < vectorSize; j++) {
            double exponent = (double) (2 * (j / 2)) / vectorSize;
            invFreq[j] = 1.0 / Math.pow(10000.0, exponent);
        }

        double[][][] inputEmbeddings = new double[batchSize][maxSeqLen][vectorSize];

        // --- バッチ対応・マルチヘッド対応させた作業スペース ---
        double[][][][][] query = new double[batchSize][numLayers][maxSeqLen][num_heads][head_size];
        double[][][][][] key = new double[batchSize][numLayers][maxSeqLen][num_heads][head_size];
        double[][][][][] value = new double[batchSize][numLayers][maxSeqLen][num_heads][head_size];

        int attentionSize = batchSize * numLayers * num_heads * maxSeqLen * maxSeqLen;
        double[] attentionScores = new double[attentionSize];
        double[] attentionWeights = new double[attentionSize];


        double[][][][] attentionOutput = new double[batchSize][numLayers][maxSeqLen][vectorSize];
        double[][][][] ffnOutput = new double[batchSize][numLayers][maxSeqLen][vectorSize];


        String[] modenames = new String[3];
        modenames[0] = "事前学習";
        modenames[1] = "対話学習";
        modenames[2] = "ミックス学習";

        if (!larned) {


            double[][][][] dFfnOutput = new double[batchSize][numLayers][maxSeqLen][vectorSize];
            double[][][][] dAttentionOutput = new double[batchSize][numLayers][maxSeqLen][vectorSize];
            double[] dAttentionWeights = new double[attentionSize];
            double[][][][][] dValue = new double[batchSize][numLayers][maxSeqLen][num_heads][head_size];
            double[] dAttentionScores = new double[attentionSize];
            double[][][][][] dQuery = new double[batchSize][numLayers][maxSeqLen][num_heads][head_size];
            double[][][][][] dKey = new double[batchSize][numLayers][maxSeqLen][num_heads][head_size];
            double[][][] dInputEmbeddings = new double[batchSize][maxSeqLen][vectorSize];


            double[][] mRmsWeightAttention = new double[numLayers][vectorSize];
            double[][] vRmsWeightAttention = new double[numLayers][vectorSize];
            double[][] mRmsWeightFfn = new double[numLayers][vectorSize];
            double[][] vRmsWeightFfn = new double[numLayers][vectorSize];


            // --- ローカル勾配（各スレッドが自分専用に使う作業机） ---
            double[][][] localDWOut = new double[batchSize][vectorSize][vocabSize];
            double[][][][] localDWGate = new double[batchSize][numLayers][vectorSize][vectorSize * 2];
            double[][][][] localDWUp = new double[batchSize][numLayers][vectorSize][vectorSize * 2];
            double[][][][] localDWDown = new double[batchSize][numLayers][vectorSize * 2][vectorSize];
            double[][][][][] localDWq = new double[batchSize][numLayers][num_heads][vectorSize][head_size];
            double[][][][][] localDWk = new double[batchSize][numLayers][num_heads][vectorSize][head_size];
            double[][][][][] localDWv = new double[batchSize][numLayers][num_heads][vectorSize][head_size];
            double[][][] localDEmbeddingTable = new double[batchSize][vocabSize][vectorSize];

            double[][][] localDRmsWeightAttention = new double[batchSize][numLayers][vectorSize];
            double[][][] localDRmsWeightFfn = new double[batchSize][numLayers][vectorSize];


            double[][] globalDRmsWeightAttention = new double[numLayers][vectorSize];
            double[][] globalDRmsWeightFfn = new double[numLayers][vectorSize];

            double[][][] mDWGate = new double[numLayers][vectorSize][dHidden];
            double[][][] vDWGate = new double[numLayers][vectorSize][dHidden];
            double[][][] mDWUp = new double[numLayers][vectorSize][dHidden];
            double[][][] vDWUp = new double[numLayers][vectorSize][dHidden];
            double[][][] mDWDown = new double[numLayers][dHidden][vectorSize];
            double[][][] vDWDown = new double[numLayers][dHidden][vectorSize];

            // --- AdamW用のモーメント配列 ---
            double[][] mDWOut = new double[vectorSize][vocabSize];
            double[][] vDWOut = new double[vectorSize][vocabSize];
            double[][] mDEmbeddingTable = new double[vocabSize][vectorSize];
            double[][] vDEmbeddingTable = new double[vocabSize][vectorSize];

            double[][][][] mDWq = new double[numLayers][num_heads][vectorSize][head_size];
            double[][][][] vDWq = new double[numLayers][num_heads][vectorSize][head_size];
            double[][][][] mDWk = new double[numLayers][num_heads][vectorSize][head_size];
            double[][][][] vDWk = new double[numLayers][num_heads][vectorSize][head_size];
            double[][][][] mDWv = new double[numLayers][num_heads][vectorSize][head_size];
            double[][][][] vDWv = new double[numLayers][num_heads][vectorSize][head_size];


            // --- グローバル勾配（全スレッドの結果を最終的に合算する場所） ---
            double[][] globalDWOut = new double[vectorSize][vocabSize];
            double[][][] globalDWGate = new double[numLayers][vectorSize][vectorSize * 2];
            double[][][] globalDWUp = new double[numLayers][vectorSize][vectorSize * 2];
            double[][][] globalDWDown = new double[numLayers][vectorSize * 2][vectorSize];
            double[][][][] globalDWq = new double[numLayers][num_heads][vectorSize][head_size];
            double[][][][] globalDWk = new double[numLayers][num_heads][vectorSize][head_size];
            double[][][][] globalDWv = new double[numLayers][num_heads][vectorSize][head_size];
            double[][] globalDEmbeddingTable = new double[vocabSize][vectorSize];


            for (int i = 0; i < vocabSize; i++) {
                for (int j = 0; j < vectorSize; j++) {
                    embeddingTable[i][j] = (random.nextDouble() - 0.5);
                }
            }

            long startTime = System.nanoTime();
            long endTime;
            double elapsedSeconds;


            System.out.println("=== 学習開始 ===");

            int epochstatus = 0;
            boolean needHeader = true;

            int epoch = 0;
            while (true) {

                String[] currentDataset;
                // フェーズごとのデータセット選択とタイトル表示
                if (epochstatus == 0) {
                    if (needHeader) {
                        IO.println("=====事前学習=====");
                        needHeader = false;
                    }
                    currentDataset = pretrainDataset;
                } else if (epochstatus == 1) {
                    if (needHeader) {
                        IO.println("=====対話学習=====");
                        needHeader = false;
                    }
                    currentDataset = chatDataset;
                } else if (epochstatus == 2) {
                    if (needHeader) {
                        IO.println("=====混合学習=====");
                        needHeader = false;
                    }
                    currentDataset = allDataset;
                } else {
                    // すべてのフェーズが完了したらループを抜ける
                    System.out.println("すべての学習フェーズが完了しました！");
                    break;
                }

                Integer[] indices = new Integer[currentDataset.length];
                for (int i = 0; i < indices.length; i++) {
                    indices[i] = i;
                }
                List<Integer> indexList = Arrays.asList(indices);
                Collections.shuffle(indexList, random);
                Integer[] shuffledIndices = indexList.toArray(new Integer[0]);

                // --- 1. グローバル勾配のゼロクリア ---
                for (double[] row : globalDWOut) Arrays.fill(row, 0.0);
                for (double[] row : globalDEmbeddingTable) Arrays.fill(row, 0.0);

                for (double[] row : globalDRmsWeightAttention) Arrays.fill(row, 0.0);
                for (double[] row : globalDRmsWeightFfn) Arrays.fill(row, 0.0);

                for (double[][] matrix : globalDWGate) for (double[] row : matrix) Arrays.fill(row, 0.0);
                for (double[][] matrix : globalDWUp) for (double[] row : matrix) Arrays.fill(row, 0.0);
                for (double[][] matrix : globalDWDown) for (double[] row : matrix) Arrays.fill(row, 0.0);
                for (double[][][] tensor : globalDWq)
                    for (double[][] matrix : tensor) for (double[] row : matrix) Arrays.fill(row, 0.0);
                for (double[][][] tensor : globalDWk)
                    for (double[][] matrix : tensor) for (double[] row : matrix) Arrays.fill(row, 0.0);
                for (double[][][] tensor : globalDWv)
                    for (double[][] matrix : tensor) for (double[] row : matrix) Arrays.fill(row, 0.0);

                // --- 2. 各スレッド（ローカル）の勾配もゼロクリア ---
                for (int b = 0; b < batchSize; b++) {
                    for (double[] row : localDWOut[b]) Arrays.fill(row, 0.0);
                    for (double[] row : localDEmbeddingTable[b]) Arrays.fill(row, 0.0);

                    for (double[] row : localDRmsWeightAttention[b]) Arrays.fill(row, 0.0);
                    for (double[] row : localDRmsWeightFfn[b]) Arrays.fill(row, 0.0);

                    for (double[][] matrix : localDWGate[b]) for (double[] row : matrix) Arrays.fill(row, 0.0);
                    for (double[][] matrix : localDWUp[b]) for (double[] row : matrix) Arrays.fill(row, 0.0);
                    for (double[][] matrix : localDWDown[b]) for (double[] row : matrix) Arrays.fill(row, 0.0);
                    for (double[][][] tensor : localDWq[b])
                        for (double[][] matrix : tensor) for (double[] row : matrix) Arrays.fill(row, 0.0);
                    for (double[][][] tensor : localDWk[b])
                        for (double[][] matrix : tensor) for (double[] row : matrix) Arrays.fill(row, 0.0);
                    for (double[][][] tensor : localDWv[b])
                        for (double[][] matrix : tensor) for (double[] row : matrix) Arrays.fill(row, 0.0);
                }


                // バッチ全体の損失を記録する用（スレッドセーフな加算器）
                java.util.concurrent.atomic.DoubleAdder epochTotalLoss = new java.util.concurrent.atomic.DoubleAdder();

                // --- 3. 16個のデータを並列（ParallelStream）で同時に処理 ---
                final int currentEpoch = epoch;
                final String[] finalCurrentDataset = currentDataset;
                final Integer[] finalShuffledIndices = shuffledIndices;

                java.util.stream.IntStream.range(0, batchSize).parallel().forEach(b -> {
                    String currentText;

                    // ★ シャッフルされた配列から順番にデータを取得する
                    int dataIndex = finalShuffledIndices[b % finalShuffledIndices.length];
                    currentText = finalCurrentDataset[dataIndex];

                    List<Integer> encodedList = tokenizer.encode(currentText);
                    int seqLen = encodedList.size();
                    int[] encoded = new int[seqLen];
                    for (int i = 0; i < seqLen; i++) {
                        encoded[i] = encodedList.get(i);
                    }


                    boolean isChatData = currentText.startsWith("U:");
                    int startPredictIdx = 0;

                    if (isChatData) {
                        int aIndex = currentText.indexOf(">A:");
                        if (aIndex != -1) {
                            String promptPart = currentText.substring(0, aIndex + 3);
                            List<Integer> promptEncoded = tokenizer.encode(promptPart);
                            startPredictIdx = promptEncoded.size() - 1;
                        }
                    }


                    // 4. Embedding Lookup
                    for (int i = 0; i < seqLen; i++) {
                        int id = encoded[i];
                        System.arraycopy(embeddingTable[id], 0, inputEmbeddings[b][i], 0, vectorSize);
                    }

                    // 5. Positional Encoding
                    for (int i = 0; i < seqLen; i++) {
                        for (int j = 0; j < vectorSize; j++) {
                            double angle = i * invFreq[j];
                            if (j % 2 == 0) {
                                inputEmbeddings[b][i][j] += Math.sin(angle);
                            } else {
                                inputEmbeddings[b][i][j] += Math.cos(angle);
                            }
                        }
                    }


                    // --- 順伝播 ---
                    for (int l = 0; l < numLayers; l++) {
                        final int layer = l;
                        double[][] currentLayerInput = (layer == 0) ? inputEmbeddings[b] : ffnOutput[b][layer - 1];

                        // ★追加: Attentionに入る前にRMSNormをかける
                        double[][] normedInput = new double[seqLen][dModel];
                        for (int i = 0; i < seqLen; i++) {
                            normedInput[i] = rmsNorm(currentLayerInput[i], rmsWeightAttention[layer], 1e-5);
                        }

                        // 6. Query (Q) ※ currentLayerInput の代わりに normedInput を使う
                        for (int h = 0; h < num_heads; h++) {
                            for (int i = 0; i < seqLen; i++) {
                                for (int j = 0; j < head_size; j++) {
                                    double sum = 0.0;
                                    for (int k = 0; k < dModel; k++) {
                                        sum += normedInput[i][k] * wq[layer][h][k][j]; // ← normedInputに変更
                                    }
                                    query[b][layer][i][h][j] = sum;
                                }
                            }
                        }

                        // 7. Key (K) も同様に normedInput を使う
                        for (int h = 0; h < num_heads; h++) {
                            for (int i = 0; i < seqLen; i++) {
                                for (int j = 0; j < head_size; j++) {
                                    double sum = 0.0;
                                    for (int k = 0; k < dModel; k++) {
                                        sum += normedInput[i][k] * wk[layer][h][k][j]; // ← normedInputに変更
                                    }
                                    key[b][layer][i][h][j] = sum;
                                }
                            }
                        }

                        // 8. Value (V) も同様に normedInput を使う
                        for (int h = 0; h < num_heads; h++) {
                            for (int i = 0; i < seqLen; i++) {
                                for (int j = 0; j < head_size; j++) {
                                    double sum = 0.0;
                                    for (int k = 0; k < dModel; k++) {
                                        sum += normedInput[i][k] * wv[layer][h][k][j]; // ← normedInputに変更
                                    }
                                    value[b][layer][i][h][j] = sum;
                                }
                            }
                        }

                        // 9. Attention スコア（ヘッドごとに計算）
                        double scale = Math.sqrt(head_size);

                        for (int h = 0; h < num_heads; h++) { // ヘッドのループを追加
                            for (int i = 0; i < seqLen; i++) {
                                int startIndex = getAttentionIndex(b, layer, h, i, 0, numLayers, num_heads, maxSeqLen);

                                Arrays.fill(attentionScores, startIndex, startIndex + seqLen, -1e9);
                                for (int j = 0; j <= i; j++) {
                                    double dotProduct = 0.0;
                                    for (int k = 0; k < head_size; k++) { // 内積はヘッド内の次元（head_size）で計算
                                        dotProduct += query[b][layer][i][h][k] * key[b][layer][j][h][k];
                                    }
                                    attentionScores[startIndex + j] = dotProduct / scale;
                                }
                            }
                        }

                        // 10. Softmax（ヘッドごとに独立して計算）
                        for (int h = 0; h < num_heads; h++) {
                            for (int i = 0; i < seqLen; i++) {
                                // インデックスを計算
                                int startIndex = getAttentionIndex(b, layer, h, i, 0, numLayers, num_heads, seqLen);

                                // maxの取得（j = 0）
                                double max = attentionScores[startIndex];
                                for (int j = 1; j < seqLen; j++) {
                                    double score = attentionScores[startIndex + j];
                                    if (score > max) {
                                        max = score;
                                    }
                                }

                                double sum = 0.0;
                                double[] expRow = new double[seqLen];
                                for (int j = 0; j < seqLen; j++) {
                                    expRow[j] = Math.exp(attentionScores[startIndex + j] - max);
                                    sum += expRow[j];
                                }

                                for (int j = 0; j < seqLen; j++) {
                                    // attentionWeights も1次元化しているので startIndex + j を使う
                                    attentionWeights[startIndex + j] = expRow[j] / sum;
                                }
                            }
                        }

                        // 11. Attention Output（各ヘッドの結果を Concatenate して元の次元に戻す ＋ 残差接続）
                        for (int i = 0; i < seqLen; i++) {
                            int outCol = 0; // 結合していくときのインデックス用
                            for (int h = 0; h < num_heads; h++) {
                                // ★ここで h ごとの startIndex を計算
                                int startIndex = getAttentionIndex(b, layer, h, i, 0, numLayers, num_heads, seqLen);

                                for (int j = 0; j < head_size; j++) {
                                    double weightedSum = 0.0;
                                    for (int k = 0; k < seqLen; k++) {
                                        weightedSum += attentionWeights[startIndex + k] * value[b][layer][k][h][j];
                                    }
                                    // 各ヘッドの計算結果を横に繋ぎ合わせていく（Concat）
                                    attentionOutput[b][layer][i][outCol++] = weightedSum;
                                }
                            }
                            // 残差接続（Residual Connection）：入力層のベクトルをそのまま足し合わせる
                            for (int j = 0; j < vectorSize; j++) {
                                attentionOutput[b][layer][i][j] += currentLayerInput[i][j];
                            }
                        }

                        double[][] normedAttentionOutput = new double[seqLen][dModel];
                        for (int i = 0; i < seqLen; i++) {
                            normedAttentionOutput[i] = rmsNorm(attentionOutput[b][layer][i], rmsWeightFfn[layer], 1e-5);
                        }

                        // 12. FFN
                        for (int i = 0; i < seqLen; i++) {
                            double[] gate = new double[dHidden];
                            double[] up = new double[dHidden];
                            double[] gatedValue = new double[dHidden];

                            for (int j = 0; j < dHidden; j++) {
                                double sumGate = 0.0;
                                double sumUp = 0.0;
                                for (int k = 0; k < dModel; k++) {
                                    double val = normedAttentionOutput[i][k]; // ← normedAttentionOutputを使う
                                    sumGate += val * wGate[layer][k][j];
                                    sumUp += val * wUp[layer][k][j];
                                }
                                double sigmoid = 1.0 / (1.0 + Math.exp(-sumGate));
                                gate[j] = sumGate * sigmoid;
                                up[j] = sumUp;
                            }

                            for (int j = 0; j < dHidden; j++) {
                                gatedValue[j] = gate[j] * up[j];
                            }

                            // FFNの最後でも「FFNの出力 ＋ Attention出力（残差接続）」をします
                            for (int j = 0; j < dModel; j++) {
                                double sum = 0.0;
                                for (int k = 0; k < dHidden; k++) {
                                    sum += gatedValue[k] * wDown[layer][k][j];
                                }
                                ffnOutput[b][layer][i][j] = sum + attentionOutput[b][layer][i][j]; // 残差接続
                            }
                        }
                    }

                    // 13. 逆伝播ワークスペース初期化
                    for (int l = 0; l < numLayers; l++) {
                        for (int i = 0; i < seqLen; i++) {
                            Arrays.fill(dFfnOutput[b][l][i], 0.0);
                            Arrays.fill(dAttentionOutput[b][l][i], 0.0);
                        }
                        // 【修正】ヘッド数を含む配列の初期化
                        for (int h = 0; h < num_heads; h++) {
                            for (int i = 0; i < seqLen; i++) {
                                Arrays.fill(dQuery[b][l][i][h], 0.0);
                                Arrays.fill(dKey[b][l][i][h], 0.0);
                                Arrays.fill(dValue[b][l][i][h], 0.0);
                            }
                            for (int i = 0; i < seqLen; i++) {
                                int startIndex = getAttentionIndex(b, l, h, i, 0, numLayers, num_heads, seqLen);

                                Arrays.fill(dAttentionWeights, startIndex, startIndex + seqLen, 0.0);
                                Arrays.fill(dAttentionScores, startIndex, startIndex + seqLen, 0.0);
                            }
                        }
                    }
                    for (int i = 0; i < seqLen; i++) {
                        Arrays.fill(dInputEmbeddings[b][i], 0.0);
                    }

                    // --- 出力層の逆伝播 & Loss計算 ---
                    int numPredictions = seqLen - 1;
                    double threadLoss = 0.0;
                    int validPredictionCount = 0;
                    for (int i = 0; i < numPredictions; i++) {
                        if (isChatData && i < startPredictIdx) {
                            continue;
                        }
                        int targetId = encoded[i + 1];
                        double[] logits = new double[vocabSizeLocal];
                        for (int j = 0; j < vocabSizeLocal; j++) {
                            double sum = 0.0;
                            for (int k = 0; k < dModel; k++) {
                                sum += ffnOutput[b][numLayers - 1][i][k] * wOut[k][j];
                            }
                            logits[j] = sum;
                        }

                        double maxLogit = logits[0];
                        for (int j = 1; j < vocabSizeLocal; j++) {
                            if (logits[j] > maxLogit) maxLogit = logits[j];
                        }

                        double[] probs = new double[vocabSizeLocal];
                        double pSum = 0.0;
                        for (int j = 0; j < vocabSizeLocal; j++) {
                            probs[j] = Math.exp(logits[j] - maxLogit);
                            pSum += probs[j];
                        }
                        for (int j = 0; j < vocabSizeLocal; j++) {
                            probs[j] /= pSum;
                        }

                        // 損失の加算
                        threadLoss += -Math.log(Math.max(probs[targetId], 1e-15));
                        validPredictionCount++; // 有効な予測数をカウントする

                        double[] dLogits = new double[vocabSizeLocal];
                        System.arraycopy(probs, 0, dLogits, 0, vocabSizeLocal);
                        dLogits[targetId] -= 1.0;

                        for (int k = 0; k < dModel; k++) {
                            for (int j = 0; j < vocabSizeLocal; j++) {
                                localDWOut[b][k][j] += ffnOutput[b][numLayers - 1][i][k] * dLogits[j];
                            }
                            double gradSum = 0.0;
                            for (int j = 0; j < vocabSizeLocal; j++) {
                                gradSum += dLogits[j] * wOut[k][j];
                            }
                            dFfnOutput[b][numLayers - 1][i][k] += gradSum;
                        }
                    }
                    if (validPredictionCount > 0) {
                        epochTotalLoss.add(threadLoss / validPredictionCount);
                    } else {
                        epochTotalLoss.add(0.0);
                        IO.println("INFO: epochTotalLossが0文字でした");
                    }

                    // --- 各レイヤーの逆伝播 ---
                    for (int l = numLayers - 1; l >= 0; l--) {
                        double[][] currentLayerInput = (l == 0) ? inputEmbeddings[b] : ffnOutput[b][l - 1];

                        for (int i = 0; i < seqLen; i++) {
                            for (int k = 0; k < dModel; k++) {
                                dAttentionOutput[b][l][i][k] += dFfnOutput[b][l][i][k];
                            }
                        }

                        // FFN逆伝播
                        for (int i = 0; i < seqLen; i++) {
                            double[] sumGateArr = new double[dHidden];
                            double[] sigmoidArr = new double[dHidden];
                            double[] gate = new double[dHidden];
                            double[] up = new double[dHidden];

                            for (int j = 0; j < dHidden; j++) {
                                double sum = 0.0;
                                for (int k = 0; k < dModel; k++) {
                                    sum += attentionOutput[b][l][i][k] * wGate[l][k][j];
                                }
                                sumGateArr[j] = sum;
                                double sigmoid = 1.0 / (1.0 + Math.exp(-sum));
                                sigmoidArr[j] = sigmoid;
                                gate[j] = sum * sigmoid;
                            }

                            for (int j = 0; j < dHidden; j++) {
                                double sum = 0.0;
                                for (int k = 0; k < dModel; k++) {
                                    sum += attentionOutput[b][l][i][k] * wUp[l][k][j];
                                }
                                up[j] = sum;
                            }

                            double[] gatedValue = new double[dHidden];
                            for (int j = 0; j < dHidden; j++) {
                                gatedValue[j] = gate[j] * up[j];
                            }

                            double[] dGatedValue = new double[dHidden];
                            for (int k = 0; k < dHidden; k++) {
                                double sum = 0.0;
                                for (int j = 0; j < dModel; j++) {
                                    sum += dFfnOutput[b][l][i][j] * wDown[l][k][j];
                                    localDWDown[b][l][k][j] += gatedValue[k] * dFfnOutput[b][l][i][j];
                                }
                                dGatedValue[k] = sum;
                            }

                            for (int j = 0; j < dHidden; j++) {
                                double dGate_j = dGatedValue[j] * up[j];
                                double dUp_j = dGatedValue[j] * gate[j];
                                double sum = sumGateArr[j];
                                double s = sigmoidArr[j];
                                double dSilu = s + (sum * s) * (1.0 - s);
                                double dSumGate = dGate_j * dSilu;

                                for (int k = 0; k < dModel; k++) {
                                    localDWGate[b][l][k][j] += attentionOutput[b][l][i][k] * dSumGate;
                                    localDWUp[b][l][k][j] += attentionOutput[b][l][i][k] * dUp_j;
                                    dAttentionOutput[b][l][i][k] += dSumGate * wGate[l][k][j] + dUp_j * wUp[l][k][j];
                                }
                            }
                        }


                        // ==========================================
                        // FFN前のRMSNormの逆伝播
                        // ==========================================
                        for (int i = 0; i < seqLen; i++) {
                            double[] x = attentionOutput[b][l][i]; // RMSNormの入力（生の値）
                            double[] dyNormed = new double[dModel]; // FFNから伝わってきた勾配
                            for (int k = 0; k < dModel; k++) {
                                dyNormed[k] = dAttentionOutput[b][l][i][k];
                                dAttentionOutput[b][l][i][k] = 0.0; // 一旦リセットして上書き
                            }

                            // 分散とRMSの再計算
                            double sumSq = 0.0;
                            for (int k = 0; k < dModel; k++) {
                                sumSq += x[k] * x[k];
                            }
                            double rms = Math.sqrt((sumSq / dModel) + 1e-5);
                            double invRms = 1.0 / rms;

                            // 1. RMSNormの重み（rmsWeightFfn）の勾配を計算
                            for (int k = 0; k < dModel; k++) {
                                double normedVal = x[k] * invRms;
                                localDRmsWeightFfn[b][l][k] += dyNormed[k] * normedVal;
                            }

                            // 2. 入力 x に対する勾配（dx）を計算して dAttentionOutput に戻す
                            double sumTerm = 0.0;
                            for (int k = 0; k < dModel; k++) {
                                sumTerm += dyNormed[k] * rmsWeightFfn[l][k] * (x[k] * invRms);
                            }

                            for (int k = 0; k < dModel; k++) {
                                double term1 = dyNormed[k] * rmsWeightFfn[l][k];
                                double term2 = (x[k] * invRms) * sumTerm / dModel;
                                dAttentionOutput[b][l][i][k] = invRms * (term1 - term2);
                            }
                        }

// 1. Attention Output から Attention Weights と Value への勾配（Concatをバラす）
                        for (int i = 0; i < seqLen; i++) {
                            for (int h = 0; h < num_heads; h++) {
                                // ★ここで h と i に応じた startIndex を取得
                                int startIndex = getAttentionIndex(b, l, h, i, 0, numLayers, num_heads, seqLen);

                                for (int j = 0; j < head_size; j++) {
                                    int outCol = h * head_size + j;
                                    double gradOut = dAttentionOutput[b][l][i][outCol];
                                    for (int k = 0; k < seqLen; k++) {
                                        // ★1次元のインデックス（startIndex + k）に書き換え
                                        dAttentionWeights[startIndex + k] += gradOut * value[b][l][k][h][j];
                                        dValue[b][l][k][h][j] += gradOut * attentionWeights[startIndex + k];
                                    }
                                }
                            }
                        }

                        // 2. Softmax の逆伝播（ヘッドごとに計算）
                        for (int h = 0; h < num_heads; h++) {
                            for (int i = 0; i < seqLen; i++) {
                                // ★startIndex を取得
                                int startIndex = getAttentionIndex(b, l, h, i, 0, numLayers, num_heads, seqLen);

                                double dotSumWeights = 0.0;
                                for (int k = 0; k < seqLen; k++) {
                                    // ★startIndex + k でアクセス
                                    dotSumWeights += dAttentionWeights[startIndex + k] * attentionWeights[startIndex + k];
                                }

                                for (int j = 0; j < seqLen; j++) {
                                    // ★startIndex + j でアクセス
                                    double y_j = attentionWeights[startIndex + j];
                                    dAttentionScores[startIndex + j] = y_j * (dAttentionWeights[startIndex + j] - dotSumWeights);
                                }
                            }

                            // マスク処理（未来の情報を隠すための因果マスクの逆伝播対応）
                            for (int i = 0; i < seqLen; i++) {
                                // ★ここでも i ごとに startIndex を取得
                                int startIndex = getAttentionIndex(b, l, h, i, 0, numLayers, num_heads, seqLen);
                                for (int j = 0; j < seqLen; j++) {
                                    // ★startIndex + j でアクセス
                                    if (j > i) dAttentionScores[startIndex + j] = 0.0;
                                }
                            }
                        }

                        // 3. Query と Key への勾配計算（ヘッドごと、scaleは head_size の平方根）
                        double scale = Math.sqrt(head_size);
                        for (int h = 0; h < num_heads; h++) {
                            for (int i = 0; i < seqLen; i++) {
                                int startIndex = getAttentionIndex(b, l, h, i, 0, numLayers, num_heads, seqLen);
                                for (int j = 0; j < seqLen; j++) {
                                    double dScore = dAttentionScores[startIndex + j] / scale;
                                    for (int k = 0; k < head_size; k++) {
                                        dQuery[b][l][i][h][k] += dScore * key[b][l][j][h][k];
                                        dKey[b][l][j][h][k] += dScore * query[b][l][i][h][k];
                                    }
                                }
                            }
                        }

                        // 4. 残差接続の勾配（Attention Output から入力側へそのまま流す）
                        for (int i = 0; i < seqLen; i++) {
                            for (int k = 0; k < vectorSize; k++) {
                                double gradOut = dAttentionOutput[b][l][i][k];
                                if (l > 0) {
                                    dFfnOutput[b][l - 1][i][k] += gradOut;
                                } else {
                                    dInputEmbeddings[b][i][k] += gradOut;
                                }
                            }
                        }

                        // 一時的にAttention側からの入力勾配を集める配列
                        double[][] dNormedInput = new double[seqLen][vectorSize];

                        // 5. Query と Value の重み勾配（wq, wv）と、入力への勾配計算
                        for (int i = 0; i < seqLen; i++) {
                            for (int h = 0; h < num_heads; h++) {
                                for (int j = 0; j < head_size; j++) {
                                    double gQ = dQuery[b][l][i][h][j];
                                    double gV = dValue[b][l][i][h][j];

                                    for (int k = 0; k < vectorSize; k++) {
                                        localDWq[b][l][h][k][j] += currentLayerInput[i][k] * gQ;
                                        localDWv[b][l][h][k][j] += currentLayerInput[i][k] * gV;

                                        // 直接前の層に足すのではなく、dNormedInputに集める
                                        dNormedInput[i][k] += gQ * wq[l][h][k][j] + gV * wv[l][h][k][j];
                                    }
                                }
                            }
                        }

                        // 6. Key の重み勾配（wk）と、入力への勾配計算
                        for (int j = 0; j < seqLen; j++) {
                            for (int h = 0; h < num_heads; h++) {
                                for (int k = 0; k < head_size; k++) {
                                    double gK = dKey[b][l][j][h][k];
                                    for (int m = 0; m < vectorSize; m++) {
                                        localDWk[b][l][h][m][k] += currentLayerInput[j][m] * gK;

                                        // ここもdNormedInputに集める
                                        dNormedInput[j][m] += gK * wk[l][h][m][k];
                                    }
                                }
                            }
                        }

                        // ==========================================
                        // 【追加】Attention前のRMSNormの逆伝播
                        // ==========================================
                        for (int i = 0; i < seqLen; i++) {
                            double[] x = currentLayerInput[i]; // RMSNormを通る前の生の値
                            double[] dyNormed = dNormedInput[i]; // Attentionから伝わってきた勾配

                            // 分散とRMSの再計算
                            double sumSq = 0.0;
                            for (int k = 0; k < vectorSize; k++) {
                                sumSq += x[k] * x[k];
                            }
                            double rms = Math.sqrt((sumSq / vectorSize) + 1e-5);
                            double invRms = 1.0 / rms;

                            // 1. RMSNormの重み（rmsWeightAttention）の勾配を計算
                            for (int k = 0; k < vectorSize; k++) {
                                double normedVal = x[k] * invRms;
                                localDRmsWeightAttention[b][l][k] += dyNormed[k] * normedVal;
                            }

                            // 2. 入力 x に対する勾配を計算して、前の層（dFfnOutput または dInputEmbeddings）へ加算
                            double sumTerm = 0.0;
                            for (int k = 0; k < vectorSize; k++) {
                                sumTerm += dyNormed[k] * rmsWeightAttention[l][k] * (x[k] * invRms);
                            }

                            for (int k = 0; k < vectorSize; k++) {
                                double term1 = dyNormed[k] * rmsWeightAttention[l][k];
                                double term2 = (x[k] * invRms) * sumTerm / vectorSize;
                                double gradInputVal = invRms * (term1 - term2);

                                if (l > 0) {
                                    dFfnOutput[b][l - 1][i][k] += gradInputVal;
                                } else {
                                    dInputEmbeddings[b][i][k] += gradInputVal;
                                }
                            }
                        }

// 7. 埋め込み層への勾配蓄積
                        for (int i = 0; i < seqLen; i++) {
                            int id = encoded[i];
                            for (int j = 0; j < vectorSize; j++) {
                                localDEmbeddingTable[b][id][j] += dInputEmbeddings[b][i][j];
                            }
                        }
                    }
                });
// --- 4. ローカル勾配をグローバル勾配に合算する ---
                for (int b = 0; b < batchSize; b++) {
                    for (int i = 0; i < vectorSize; i++) {
                        for (int j = 0; j < vocabSize; j++) {
                            globalDWOut[i][j] += localDWOut[b][i][j];
                        }
                    }
                    for (int i = 0; i < vocabSize; i++) {
                        for (int j = 0; j < vectorSize; j++) {
                            globalDEmbeddingTable[i][j] += localDEmbeddingTable[b][i][j];
                        }
                    }
                    for (int l = 0; l < numLayers; l++) {
                        // Q, K, V の合算をヘッド数と head_size に対応させる
                        for (int h = 0; h < num_heads; h++) {
                            for (int i = 0; i < vectorSize; i++) {
                                for (int j = 0; j < head_size; j++) {
                                    globalDWq[l][h][i][j] += localDWq[b][l][h][i][j];
                                    globalDWk[l][h][i][j] += localDWk[b][l][h][i][j];
                                    globalDWv[l][h][i][j] += localDWv[b][l][h][i][j];
                                }
                            }
                        }

                        // ==========================================
                        // RMSNormの重み勾配をグローバルに合算
                        // ==========================================
                        for (int i = 0; i < vectorSize; i++) {
                            globalDRmsWeightAttention[l][i] += localDRmsWeightAttention[b][l][i];
                            globalDRmsWeightFfn[l][i] += localDRmsWeightFfn[b][l][i];
                        }


                        for (int i = 0; i < vectorSize; i++) {
                            for (int j = 0; j < dHidden; j++) {
                                globalDWGate[l][i][j] += localDWGate[b][l][i][j];
                                globalDWUp[l][i][j] += localDWUp[b][l][i][j];
                            }
                        }
                        for (int i = 0; i < dHidden; i++) {
                            for (int j = 0; j < vectorSize; j++) {
                                globalDWDown[l][i][j] += localDWDown[b][l][i][j];
                            }
                        }
                    }
                }

                // ★ 追加：合算したグローバル勾配をバッチサイズで割って平均にする
                double invBatchSize = 1.0 / batchSize;
                for (int i = 0; i < vectorSize; i++) {
                    for (int j = 0; j < vocabSize; j++) {
                        globalDWOut[i][j] *= invBatchSize;
                    }
                }
                for (int i = 0; i < vocabSize; i++) {
                    for (int j = 0; j < vectorSize; j++) {
                        globalDEmbeddingTable[i][j] *= invBatchSize;
                    }
                }
                for (int l = 0; l < numLayers; l++) {
                    // Q, K, V の平均化もヘッド数と head_size に対応させる
                    for (int h = 0; h < num_heads; h++) {
                        for (int i = 0; i < vectorSize; i++) {
                            for (int j = 0; j < head_size; j++) {
                                globalDWq[l][h][i][j] *= invBatchSize;
                                globalDWk[l][h][i][j] *= invBatchSize;
                                globalDWv[l][h][i][j] *= invBatchSize;
                            }
                        }
                    }

                    // ==========================================
                    // 【追加】RMSNormの重み勾配の平均化
                    // ==========================================
                    for (int i = 0; i < vectorSize; i++) {
                        globalDRmsWeightAttention[l][i] *= invBatchSize;
                        globalDRmsWeightFfn[l][i] *= invBatchSize;
                    }

                    for (int i = 0; i < vectorSize; i++) {
                        for (int j = 0; j < dHidden; j++) {
                            globalDWGate[l][i][j] *= invBatchSize;
                            globalDWUp[l][i][j] *= invBatchSize;
                        }
                    }
                    for (int i = 0; i < dHidden; i++) {
                        for (int j = 0; j < vectorSize; j++) {
                            globalDWDown[l][i][j] *= invBatchSize;
                        }
                    }
                }

                // --- 5. AdamWによるパラメータ更新 ---
                adamStep++;
                double correction1 = 1.0 - Math.pow(beta1, adamStep);
                double correction2 = 1.0 - Math.pow(beta2, adamStep);

                // 1. wOut の更新
                for (int i = 0; i < dModel; i++) {
                    for (int j = 0; j < vocabSizeLocal; j++) {
                        double g = globalDWOut[i][j];
                        mDWOut[i][j] = beta1 * mDWOut[i][j] + (1.0 - beta1) * g;
                        vDWOut[i][j] = beta2 * vDWOut[i][j] + (1.0 - beta2) * (g * g);

                        double mHat = mDWOut[i][j] / correction1;
                        double vHat = vDWOut[i][j] / correction2;

                        // AdamW: Weight Decay を直接適用しつつ更新
                        wOut[i][j] = wOut[i][j] - learningRate * weightDecay * wOut[i][j] - learningRate * mHat / (Math.sqrt(vHat) + eps);
                    }
                }

                // 2. embeddingTable の更新
                for (int i = 0; i < vocabSize; i++) {
                    for (int j = 0; j < vectorSize; j++) {
                        double g = globalDEmbeddingTable[i][j];
                        mDEmbeddingTable[i][j] = beta1 * mDEmbeddingTable[i][j] + (1.0 - beta1) * g;
                        vDEmbeddingTable[i][j] = beta2 * vDEmbeddingTable[i][j] + (1.0 - beta2) * (g * g);

                        double mHat = mDEmbeddingTable[i][j] / correction1;
                        double vHat = vDEmbeddingTable[i][j] / correction2;

                        embeddingTable[i][j] = embeddingTable[i][j] - learningRate * weightDecay * embeddingTable[i][j] - learningRate * mHat / (Math.sqrt(vHat) + eps);
                    }
                }

                // 3. 各レイヤーの重み (wq, wk, wv, wGate, wUp, wDown) の更新
                for (int l = 0; l < numLayers; l++) {
                    // wq の更新（ヘッド数と head_size に対応）
                    for (int h = 0; h < num_heads; h++) {
                        for (int i = 0; i < vectorSize; i++) {
                            for (int j = 0; j < head_size; j++) {
                                double g = globalDWq[l][h][i][j];
                                mDWq[l][h][i][j] = beta1 * mDWq[l][h][i][j] + (1.0 - beta1) * g;
                                vDWq[l][h][i][j] = beta2 * vDWq[l][h][i][j] + (1.0 - beta2) * (g * g);
                                double mHat = mDWq[l][h][i][j] / correction1;
                                double vHat = vDWq[l][h][i][j] / correction2;
                                wq[l][h][i][j] = wq[l][h][i][j] - learningRate * weightDecay * wq[l][h][i][j] - learningRate * mHat / (Math.sqrt(vHat) + eps);
                            }
                        }
                    }

                    // wk の更新（ヘッド数と head_size に対応）
                    for (int h = 0; h < num_heads; h++) {
                        for (int i = 0; i < vectorSize; i++) {
                            for (int j = 0; j < head_size; j++) {
                                double g = globalDWk[l][h][i][j];
                                mDWk[l][h][i][j] = beta1 * mDWk[l][h][i][j] + (1.0 - beta1) * g;
                                vDWk[l][h][i][j] = beta2 * vDWk[l][h][i][j] + (1.0 - beta2) * (g * g);
                                double mHat = mDWk[l][h][i][j] / correction1;
                                double vHat = vDWk[l][h][i][j] / correction2;
                                wk[l][h][i][j] = wk[l][h][i][j] - learningRate * weightDecay * wk[l][h][i][j] - learningRate * mHat / (Math.sqrt(vHat) + eps);
                            }
                        }
                    }

                    // wv の更新（ヘッド数と head_size に対応）
                    for (int h = 0; h < num_heads; h++) {
                        for (int i = 0; i < vectorSize; i++) {
                            for (int j = 0; j < head_size; j++) {
                                double g = globalDWv[l][h][i][j];
                                mDWv[l][h][i][j] = beta1 * mDWv[l][h][i][j] + (1.0 - beta1) * g;
                                vDWv[l][h][i][j] = beta2 * vDWv[l][h][i][j] + (1.0 - beta2) * (g * g);
                                double mHat = mDWv[l][h][i][j] / correction1;
                                double vHat = vDWv[l][h][i][j] / correction2;
                                wv[l][h][i][j] = wv[l][h][i][j] - learningRate * weightDecay * wv[l][h][i][j] - learningRate * mHat / (Math.sqrt(vHat) + eps);
                            }
                        }
                    }

                    // wGate
                    for (int i = 0; i < dModel; i++) {
                        for (int j = 0; j < dHidden; j++) {
                            double g = globalDWGate[l][i][j];
                            mDWGate[l][i][j] = beta1 * mDWGate[l][i][j] + (1.0 - beta1) * g;
                            vDWGate[l][i][j] = beta2 * vDWGate[l][i][j] + (1.0 - beta2) * (g * g);
                            double mHat = mDWGate[l][i][j] / correction1;
                            double vHat = vDWGate[l][i][j] / correction2;
                            wGate[l][i][j] = wGate[l][i][j] - learningRate * weightDecay * wGate[l][i][j] - learningRate * mHat / (Math.sqrt(vHat) + eps);
                        }
                    }
                    // wUp
                    for (int i = 0; i < dModel; i++) {
                        for (int j = 0; j < dHidden; j++) {
                            double g = globalDWUp[l][i][j];
                            mDWUp[l][i][j] = beta1 * mDWUp[l][i][j] + (1.0 - beta1) * g;
                            vDWUp[l][i][j] = beta2 * vDWUp[l][i][j] + (1.0 - beta2) * (g * g);
                            double mHat = mDWUp[l][i][j] / correction1;
                            double vHat = vDWUp[l][i][j] / correction2;
                            wUp[l][i][j] = wUp[l][i][j] - learningRate * weightDecay * wUp[l][i][j] - learningRate * mHat / (Math.sqrt(vHat) + eps);
                        }
                    }
                    // wDown
                    for (int i = 0; i < dHidden; i++) {
                        for (int j = 0; j < dModel; j++) {
                            double g = globalDWDown[l][i][j];
                            mDWDown[l][i][j] = beta1 * mDWDown[l][i][j] + (1.0 - beta1) * g;
                            vDWDown[l][i][j] = beta2 * vDWDown[l][i][j] + (1.0 - beta2) * (g * g);
                            double mHat = mDWDown[l][i][j] / correction1;
                            double vHat = vDWDown[l][i][j] / correction2;
                            wDown[l][i][j] = wDown[l][i][j] - learningRate * weightDecay * wDown[l][i][j] - learningRate * mHat / (Math.sqrt(vHat) + eps);
                        }
                    }

                    // 4. 各レイヤーの RMSNorm の重み (rmsWeightAttention, rmsWeightFfn) の更新
                    for (int i = 0; i < vectorSize; i++) {
                        // Attention側のRMSNorm重み更新
                        double gAttn = globalDRmsWeightAttention[l][i];
                        mRmsWeightAttention[l][i] = beta1 * mRmsWeightAttention[l][i] + (1.0 - beta1) * gAttn;
                        vRmsWeightAttention[l][i] = beta2 * vRmsWeightAttention[l][i] + (1.0 - beta2) * (gAttn * gAttn);
                        double mHatAttn = mRmsWeightAttention[l][i] / correction1;
                        double vHatAttn = vRmsWeightAttention[l][i] / correction2;
                        rmsWeightAttention[l][i] = rmsWeightAttention[l][i] - learningRate * weightDecay * rmsWeightAttention[l][i] - learningRate * mHatAttn / (Math.sqrt(vHatAttn) + eps);

                        // FFN側のRMSNorm重み更新
                        double gFfn = globalDRmsWeightFfn[l][i];
                        mRmsWeightFfn[l][i] = beta1 * mRmsWeightFfn[l][i] + (1.0 - beta1) * gFfn;
                        vRmsWeightFfn[l][i] = beta2 * vRmsWeightFfn[l][i] + (1.0 - beta2) * (gFfn * gFfn);
                        double mHatFfn = mRmsWeightFfn[l][i] / correction1;
                        double vHatFfn = vRmsWeightFfn[l][i] / correction2;
                        rmsWeightFfn[l][i] = rmsWeightFfn[l][i] - learningRate * weightDecay * rmsWeightFfn[l][i] - learningRate * mHatFfn / (Math.sqrt(vHatFfn) + eps);
                    }
                }

                endTime = System.nanoTime();
                elapsedSeconds = (endTime - startTime) / 1_000_000_000.0;


                double avgLoss = epochTotalLoss.sum() / batchSize;

                IO.println("Epoch " + epoch + " - Loss: " + avgLoss + " Mode: " + modenames[epochstatus] + " 経過時間: " + elapsedSeconds + "秒");


                if (avgLoss < stopLoss) {
                    epochstatus++;
                    needHeader = true; // 次のフェーズのタイトルを出すためにフラグを戻す
                    System.out.println("ロスが" + stopLoss + "を下回ったため、次の学習データに進みます");
                }

                epoch++;
            }

            if (modelSavePath != null) {
                try {
                    ModelCheckpoint.saveModel(modelSavePath, numLayers, vectorSize, num_heads, dHidden, vocabSize, tokenizer.tokenToId, embeddingTable, wq, wk, wv, wGate, wUp, wDown, wOut, rmsWeightAttention, rmsWeightFfn, tokenizer.merges, omomilearnusedatamoto);
                } catch (IOException e) {
                    e.printStackTrace();
                    IO.println("重みの保存に失敗しました!");
                }
            }


            IO.println("学習完了!");


            System.gc(); //一応GC呼んでみる
        }

// --- 対話・文字生成テスト（KVキャッシュ対応版） ---
        IO.println();
        System.out.println("=====チャット=====");
        IO.println("重み学習に使用したデータ元: " + omomilearnusedatamoto);
        IO.println(" Hint: /helpでコマンド一覧と使い方を表示");
        IO.println();

        double[] logits = new double[vocabSizeLocal];
        double[] probs = new double[vocabSizeLocal];
        Integer[] vocabIndices = new Integer[vocabSizeLocal];
        for (int j = 0; j < vocabSizeLocal; j++) {
            vocabIndices[j] = j;
        }
        boolean[] keep = new boolean[vocabSizeLocal];

        // 推論用のKVキャッシュ（レイヤー、ヘッド、最大長、ヘッドサイズ）
        double[][][][] keyCache = new double[numLayers][num_heads][maxSeqLen][head_size];
        double[][][][] valueCache = new double[numLayers][num_heads][maxSeqLen][head_size];

        List<String> history = new ArrayList<>();
        StringBuilder generatedResponse = new StringBuilder();
        int[] genEncoded = new int[maxSeqLen];

        while (true) {
            System.out.print("AIへの入力文字をどうぞ: ");
            String userInput = scanner.nextLine();

            if (userInput != null && userInput.startsWith("/")) {
                if (Objects.equals(userInput, "/exit")) {
                    IO.println("終了します");
                    break;
                }
                if (Objects.equals(userInput, "/cleartalk")) {
                    history.clear();
                    IO.println("会話履歴をリセットしました");
                    continue;
                }
                if (Objects.equals(userInput, "/subwords")) {
                    System.out.println("\n作成されたサブワード:");
                    tokenizer.tokenToId.forEach((key1, value1) -> System.out.println("  [" + value1 + "] " + key1));
                    continue;
                }
                if (Objects.equals(userInput, "/help")) {
                    IO.println();
                    IO.println("=============================================");
                    IO.println("           コマンド一覧と使い方");
                    IO.println("=============================================");
                    IO.println();
                    IO.println();
                    IO.println("/help");
                    IO.println(" コマンド一覧と使い方を表示します");
                    IO.println();
                    IO.println("/exit");
                    IO.println(" 終了します");
                    IO.println();
                    IO.println("/cleartalk");
                    IO.println(" 会話履歴をリセットします");
                    IO.println();
                    IO.println("/subwords");
                    IO.println(" BPEによって作成されたサブワードを表示します");
                    IO.println();
                    IO.println();
                    IO.println("=============================================");
                    IO.println();
                }
                IO.println("不明なコマンド、/helpでコマンド一覧と使い方を表示");
                continue;
            }

            history.add("U:" + userInput);
            String prompt = String.join(">", history) + ">A:";

            List<Integer> promptEncodedList = tokenizer.encode(prompt);
            int promptLen = promptEncodedList.size();

            for (int i = 0; i < promptLen; i++) {
                genEncoded[i] = promptEncodedList.get(i);
            }
            int currentSeqLen = promptLen; // 現在のシーケンス長を管理する変数

            System.out.print("入力: " + userInput + "  生成結果: " + prompt);

            // キャッシュをすべて0でクリア
            for (int l = 0; l < numLayers; l++) {
                for (int h = 0; h < num_heads; h++) {
                    for (int i = 0; i < maxSeqLen; i++) {
                        Arrays.fill(keyCache[l][h][i], 0.0);
                        Arrays.fill(valueCache[l][h][i], 0.0);
                    }
                }
            }

            // ==========================================
            // 1. プロンプト部分の処理 (currentSeqLen を使用)
            // ==========================================
            for (int i = 0; i < currentSeqLen; i++) {
                int id = genEncoded[i];
                System.arraycopy(embeddingTable[id], 0, inputEmbeddings[0][i], 0, vectorSize);

                for (int j = 0; j < vectorSize; j++) {
                    double angle = i * invFreq[j];
                    if (j % 2 == 0) {
                        inputEmbeddings[0][i][j] += Math.sin(angle);
                    } else {
                        inputEmbeddings[0][i][j] += Math.cos(angle);
                    }
                }
            }

            for (int l = 0; l < numLayers; l++) {
                double[][] currentLayerInput = (l == 0) ? inputEmbeddings[0] : ffnOutput[0][l - 1];

                double[][] normedInput = new double[promptLen][dModel];
                for (int i = 0; i < promptLen; i++) {
                    normedInput[i] = rmsNorm(currentLayerInput[i], rmsWeightAttention[l], 1e-5);
                }

                // Q, K, V を計算してキャッシュに保存
                for (int h = 0; h < num_heads; h++) {
                    for (int i = 0; i < promptLen; i++) {
                        for (int j = 0; j < head_size; j++) {
                            double sumQ = 0.0, sumK = 0.0, sumV = 0.0;
                            for (int k = 0; k < dModel; k++) {
                                sumQ += normedInput[i][k] * wq[l][h][k][j];
                                sumK += normedInput[i][k] * wk[l][h][k][j];
                                sumV += normedInput[i][k] * wv[l][h][k][j];
                            }
                            query[0][l][i][h][j] = sumQ;
                            keyCache[l][h][i][j] = sumK;
                            valueCache[l][h][i][j] = sumV;
                        }
                    }
                }

                // Attention Score (プロンプト内)
                double scale = Math.sqrt(head_size);
                for (int h = 0; h < num_heads; h++) {
                    for (int i = 0; i < promptLen; i++) {
                        // startIndex を取得（b = 0）
                        int startIndex = getAttentionIndex(0, l, h, i, 0, numLayers, num_heads, maxSeqLen);

                        for (int j = 0; j < promptLen; j++) {
                            if (j > i) {
                                attentionScores[startIndex + j] = -1e9;
                            } else {
                                double dotProduct = 0.0;
                                for (int k = 0; k < head_size; k++) {
                                    dotProduct += query[0][l][i][h][k] * keyCache[l][h][j][k];
                                }
                                attentionScores[startIndex + j] = dotProduct / scale;
                            }
                        }
                    }
                }

                // Softmax
                for (int h = 0; h < num_heads; h++) {
                    for (int i = 0; i < promptLen; i++) {
                        int startIndex = getAttentionIndex(0, l, h, i, 0, numLayers, num_heads, maxSeqLen);

                        double max = attentionScores[startIndex];
                        for (int j = 1; j < promptLen; j++) {
                            if (attentionScores[startIndex + j] > max) max = attentionScores[startIndex + j];
                        }
                        double sum = 0.0;
                        double[] expRow = new double[promptLen];
                        for (int j = 0; j < promptLen; j++) {
                            expRow[j] = Math.exp(attentionScores[startIndex + j] - max);
                            sum += expRow[j];
                        }
                        for (int j = 0; j < promptLen; j++) {
                            attentionWeights[startIndex + j] = expRow[j] / sum;
                        }
                    }
                }

                // Attention Output & Residual
                for (int i = 0; i < promptLen; i++) {
                    int outCol = 0;
                    for (int h = 0; h < num_heads; h++) {
                        int startIndex = getAttentionIndex(0, l, h, i, 0, numLayers, num_heads, maxSeqLen);
                        for (int j = 0; j < head_size; j++) {
                            double weightedSum = 0.0;
                            for (int k = 0; k < promptLen; k++) {
                                weightedSum += attentionWeights[startIndex + k] * valueCache[l][h][k][j];
                            }
                            attentionOutput[0][l][i][outCol++] = weightedSum;
                        }
                    }
                    for (int j = 0; j < dModel; j++) {
                        attentionOutput[0][l][i][j] += currentLayerInput[i][j];
                    }
                }

                // FFN
                double[][] normedAttentionOutput = new double[promptLen][dModel];
                for (int i = 0; i < promptLen; i++) {
                    normedAttentionOutput[i] = rmsNorm(attentionOutput[0][l][i], rmsWeightFfn[l], 1e-5);
                }

                for (int i = 0; i < promptLen; i++) {
                    double[] gate = new double[dHidden];
                    double[] up = new double[dHidden];
                    for (int j = 0; j < dHidden; j++) {
                        double sumGate = 0.0, sumUp = 0.0;
                        for (int k = 0; k < dModel; k++) {
                            double val = normedAttentionOutput[i][k];
                            sumGate += val * wGate[l][k][j];
                            sumUp += val * wUp[l][k][j];
                        }
                        double sigmoid = 1.0 / (1.0 + Math.exp(-sumGate));
                        gate[j] = sumGate * sigmoid;
                        up[j] = sumUp;
                    }
                    double[] gatedValue = new double[dHidden];
                    for (int j = 0; j < dHidden; j++) {
                        gatedValue[j] = gate[j] * up[j];
                    }
                    for (int j = 0; j < dModel; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < dHidden; k++) {
                            sum += gatedValue[k] * wDown[l][k][j];
                        }
                        ffnOutput[0][l][i][j] = sum + attentionOutput[0][l][i][j];
                    }
                }
            }

// ==========================================
            // ★追加：プロンプトの最後の位置から「最初の1文字目」を予測する
            // ==========================================
            generatedResponse.setLength(0);
            int pos = promptLen - 1; // プロンプトの最後の位置

            // 最終レイヤーの出力からロジットを計算
            Arrays.fill(logits, 0.0);
            for (int j = 0; j < vocabSizeLocal; j++) {
                double sum = 0.0;
                for (int k = 0; k < dModel; k++) {
                    sum += ffnOutput[0][numLayers - 1][pos][k] * wOut[k][j];
                }
                logits[j] = sum;
            }

            // サンプリング処理を共通関数にするか、ここで1回実行
            int bestNextId = sampleToken(logits, vocabSizeLocal, vocabIndices, probs, keep, random);

            String predictedToken = tokenizer.decodeToken(bestNextId);
            System.out.print(predictedToken);
            generatedResponse.append(predictedToken);

            // 最初の生成トークンをバッファに格納し、配列の長さを更新
            genEncoded[promptLen] = bestNextId;
            currentSeqLen = promptLen + 1;
// ==========================================
            // 2. 1トークンずつ生成するループ（KVキャッシュを使用）
            // ==========================================
            int lastTokenId;

            for (int step = 0; step < 300; step++) {

                lastTokenId = genEncoded[currentSeqLen - 1];
                pos = currentSeqLen - 1; // 現在の位置

                if (lastTokenId == tokenizer.tokenToId.getOrDefault("■", -1)) {
                    break;
                }

                // 最新の1トークン分の埋め込みと位置エンコーディング（[0] ではなく [pos] に書き込む！）
                System.arraycopy(embeddingTable[lastTokenId], 0, inputEmbeddings[0][pos], 0, vectorSize);
                for (int j = 0; j < vectorSize; j++) {
                    double angle = pos * invFreq[j];
                    if (j % 2 == 0) {
                        inputEmbeddings[0][pos][j] += Math.sin(angle);
                    } else {
                        inputEmbeddings[0][pos][j] += Math.cos(angle);
                    }
                }

                for (int l = 0; l < numLayers; l++) {
                    // l == 0 のときは入力埋め込み、l > 0 のときは前のレイヤーの pos 位置の出力を参照
                    double[] currentLayerInput = (l == 0) ? inputEmbeddings[0][pos] : ffnOutput[0][l - 1][pos];
                    double[] normedInput = rmsNorm(currentLayerInput, rmsWeightAttention[l], 1e-5);

                    // 最新のQを計算し、KとVをキャッシュの [pos] の位置に直接保存
                    for (int h = 0; h < num_heads; h++) {
                        for (int j = 0; j < head_size; j++) {
                            double sumQ = 0.0, sumK = 0.0, sumV = 0.0;
                            for (int k = 0; k < dModel; k++) {
                                sumQ += normedInput[k] * wq[l][h][k][j];
                                sumK += normedInput[k] * wk[l][h][k][j];
                                sumV += normedInput[k] * wv[l][h][k][j];
                            }
                            query[0][l][pos][h][j] = sumQ; // [0] -> [pos] に修正
                            keyCache[l][h][pos][j] = sumK;
                            valueCache[l][h][pos][j] = sumV;
                        }
                    }
// アテンションスコア：位置 pos から、過去の全位置（0 〜 pos）のキャッシュを参照
                    double scale = Math.sqrt(head_size);
                    for (int h = 0; h < num_heads; h++) {
                        // ★ ここで pos 位置における行の先頭インデックスを取得します
                        int startIndex = getAttentionIndex(0, l, h, pos, 0, numLayers, num_heads, maxSeqLen);

                        for (int j = 0; j <= pos; j++) {
                            double dotProduct = 0.0;
                            for (int k = 0; k < head_size; k++) {
                                dotProduct += query[0][l][pos][h][k] * keyCache[l][h][j][k];
                            }
                            // ★ 1次元配列に書き換え
                            attentionScores[startIndex + j] = dotProduct / scale;
                        }

                        // Softmax (0 〜 pos)
                        // 先頭（j = 0）の要素をセット
                        double max = attentionScores[startIndex];
                        for (int j = 1; j <= pos; j++) {
                            double score = attentionScores[startIndex + j];
                            if (score > max) {
                                max = score;
                            }
                        }

                        double sum = 0.0;
                        double[] expRow = new double[pos + 1];
                        for (int j = 0; j <= pos; j++) {
                            expRow[j] = Math.exp(attentionScores[startIndex + j] - max);
                            sum += expRow[j];
                        }

                        for (int j = 0; j <= pos; j++) {
                            attentionWeights[startIndex + j] = expRow[j] / sum;
                        }
                    }

                    // Attention Output (キャッシュからVを取得)
                    int outCol = 0;
                    for (int h = 0; h < num_heads; h++) {
                        // h ごとの startIndex を取得
                        int startIndex = getAttentionIndex(0, l, h, pos, 0, numLayers, num_heads, maxSeqLen);

                        for (int j = 0; j < head_size; j++) {
                            double weightedSum = 0.0;
                            for (int k = 0; k <= pos; k++) {
                                // 1次元配列から取得
                                weightedSum += attentionWeights[startIndex + k] * valueCache[l][h][k][j];
                            }
                            attentionOutput[0][l][pos][outCol++] = weightedSum;
                        }
                    }

                    // 残差接続
                    for (int j = 0; j < dModel; j++) {
                        attentionOutput[0][l][pos][j] += currentLayerInput[j]; // [0] -> [pos]
                    }

                    // FFN
                    double[] normedAttentionOutput = rmsNorm(attentionOutput[0][l][pos], rmsWeightFfn[l], 1e-5); // [0] -> [pos]
                    double[] gate = new double[dHidden];
                    double[] up = new double[dHidden];
                    for (int j = 0; j < dHidden; j++) {
                        double sumGate = 0.0, sumUp = 0.0;
                        for (int k = 0; k < dModel; k++) {
                            double val = normedAttentionOutput[k];
                            sumGate += val * wGate[l][k][j];
                            sumUp += val * wUp[l][k][j];
                        }
                        double sigmoid = 1.0 / (1.0 + Math.exp(-sumGate));
                        gate[j] = sumGate * sigmoid;
                        up[j] = sumUp;
                    }

                    double[] gatedValue = new double[dHidden];
                    for (int j = 0; j < dHidden; j++) {
                        gatedValue[j] = gate[j] * up[j];
                    }

                    for (int j = 0; j < dModel; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < dHidden; k++) {
                            sum += gatedValue[k] * wDown[l][k][j];
                        }
                        ffnOutput[0][l][pos][j] = sum + attentionOutput[0][l][pos][j]; // [0] -> [pos]
                    }
                }

                // 最終レイヤーの出力から次のトークンを予測（posの位置を使用）
                Arrays.fill(logits, 0.0);
                for (int j = 0; j < vocabSizeLocal; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < dModel; k++) {
                        sum += ffnOutput[0][numLayers - 1][pos][k] * wOut[k][j];
                    }
                    logits[j] = sum;
                }

                bestNextId = sampleToken(logits, vocabSizeLocal, vocabIndices, probs, keep, random);

                predictedToken = tokenizer.decodeToken(bestNextId);

                if (predictedToken.equals("■")) {
                    break;
                }

                System.out.print(predictedToken);
                generatedResponse.append(predictedToken);

                genEncoded[currentSeqLen] = bestNextId;
                currentSeqLen++;
            }
            System.out.println();

            String aiReply = generatedResponse.toString();
            history.add("A:" + aiReply);

            if (history.size() > 6) {
                history.removeFirst();
                history.removeFirst();
            }
        }
    }

    // RMSNormを計算するメソッドの例
    public static double[] rmsNorm(double[] input, double[] weight, double eps) {
        int dim = input.length;
        double sumSq = 0.0;
        for (int i = 0; i < dim; i++) {
            sumSq += input[i] * input[i];
        }
        double rms = Math.sqrt(sumSq / dim + eps);

        double[] output = new double[dim];
        for (int i = 0; i < dim; i++) {
            output[i] = (input[i] / rms) * weight[i];
        }
        return output;
    }

    private static int sampleToken(double[] logits, int vocabSizeLocal, Integer[] vocabIndices, double[] probs, boolean[] keep, Random random) {
        double temperature = 0.7;
        Arrays.fill(probs, 0.0);
        double sumExp = 0.0;
        double maxLogit = logits[0];
        for (int j = 1; j < vocabSizeLocal; j++) {
            if (logits[j] > maxLogit) {
                maxLogit = logits[j];
            }
        }

        for (int j = 0; j < vocabSizeLocal; j++) {
            probs[j] = Math.exp((logits[j] - maxLogit) / temperature);
            sumExp += probs[j];
        }

        for (int j = 0; j < vocabSizeLocal; j++) {
            probs[j] /= sumExp;
        }

        double topP = 0.9;
        Arrays.sort(vocabIndices, (a, b) -> Double.compare(probs[b], probs[a]));

        double cumulativeProb = 0.0;
        Arrays.fill(keep, false);
        for (int j = 0; j < vocabSizeLocal; j++) {
            int idx = vocabIndices[j];
            cumulativeProb += probs[idx];
            keep[idx] = true;
            if (cumulativeProb >= topP) {
                break;
            }
        }

        double newSum = 0.0;
        for (int j = 0; j < vocabSizeLocal; j++) {
            if (!keep[j]) {
                probs[j] = 0.0;
            } else {
                newSum += probs[j];
            }
        }
        for (int j = 0; j < vocabSizeLocal; j++) {
            if (newSum > 0) {
                probs[j] /= newSum;
            }
        }

        double r = random.nextDouble();
        double cumulative = 0.0;
        for (int j = 0; j < vocabSizeLocal; j++) {
            int idx = vocabIndices[j]; // ※ソートされたインデックスから確率を累積する
            cumulative += probs[idx];
            if (r <= cumulative) {
                return idx;
            }
        }
        return vocabIndices[0];
    }

    public static int getAttentionIndex(int b, int l, int h, int i, int j,
                                        int numLayers, int numHeads, int maxSeqLen) {
        return (((b * numLayers + l) * numHeads + h) * maxSeqLen + i) * maxSeqLen + j;
    }
}