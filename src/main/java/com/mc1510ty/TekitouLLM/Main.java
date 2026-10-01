package com.mc1510ty.TekitouLLM;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;
import java.util.Scanner;
import java.util.stream.IntStream;

public class Main {
    static void main() {

        long seed = 12345L;
        Random random = new Random(seed);

        // 1. 複数の会話パターン（データセット）を用意する
        String[] dataset = {
                "こんにちは"
        };

        // 2. すべてのテキストから文字を集めて辞書を作る
        HashMap<Character, Integer> charToId = new HashMap<>();
        HashMap<Integer, Character> idToChar = new HashMap<>();

        java.util.List<Character> uniqueChars = java.util.Arrays.stream(dataset)
                .parallel()
                .flatMapToInt(String::chars)
                .mapToObj(c -> (char) c)
                .distinct()
                .sorted()
                .collect(java.util.stream.Collectors.toList());

        int nextId = 0;
        for (char c : uniqueChars) {
            charToId.put(c, nextId);
            idToChar.put(nextId, c);
            nextId++;
        }

        int vocabSize = charToId.size();
        int vectorSize = 64;
        int dModel = vectorSize;
        int dHidden = dModel * 2;
        int vocabSizeLocal = charToId.size();

        // 3. 埋め込みテーブルの初期化
        double[][] embeddingTable = new double[vocabSize][vectorSize];
        for (int i = 0; i < vocabSize; i++) {
            for (int j = 0; j < vectorSize; j++) {
                embeddingTable[i][j] = (random.nextDouble() - 0.5);
            }
        }
// 学習設定
        int epochs = 100000;
        double learningRate = 0.0005;
        int numLayers = 2; // ★ レイヤー数をここで指定します

        // ==========================================
        // ★【一気に事前確保する配列たち（ループ外・最上部）】
        // ==========================================
        // 最大想定シーケンス長（今回の「こんにちは」は5文字なので余裕を持って32などに設定）
        int maxSeqLen = 32;

        // パラメータ・勾配用の配列（多層対応のため、先頭に numLayers を追加）
        double[][] dWOut   = new double[vectorSize][vocabSize];
        double[][][] dWGate  = new double[numLayers][vectorSize][vectorSize * 2];
        double[][][] dWUp    = new double[numLayers][vectorSize][vectorSize * 2];
        double[][][] dWDown  = new double[numLayers][vectorSize * 2][vectorSize];
        double[][][] dWq     = new double[numLayers][vectorSize][vectorSize];
        double[][][] dWk     = new double[numLayers][vectorSize][vectorSize];
        double[][][] dWv     = new double[numLayers][vectorSize][vectorSize];
        double[][] dEmbeddingTable = new double[vocabSize][vectorSize];

        double[][][] wq = new double[numLayers][dModel][dModel];
        double[][][] wk = new double[numLayers][dModel][dModel];
        double[][][] wv = new double[numLayers][dModel][dModel];
        double[][][] wGate = new double[numLayers][dModel][dHidden];
        double[][][] wUp   = new double[numLayers][dModel][dHidden];
        double[][][] wDown = new double[numLayers][dHidden][dModel];
        double[][] wOut  = new double[dModel][vocabSizeLocal];

        // 重みのランダム初期化（レイヤーごとのループに拡張）
        for (int l = 0; l < numLayers; l++) {
            for (int i = 0; i < dModel; i++) {
                for (int j = 0; j < dModel; j++) {
                    wq[l][i][j] = (random.nextDouble() - 0.5) * 0.1;
                    wk[l][i][j] = (random.nextDouble() - 0.5) * 0.1;
                    wv[l][i][j] = (random.nextDouble() - 0.5) * 0.1;
                }
            }
            for (int i = 0; i < dModel; i++) {
                for (int j = 0; j < dHidden; j++) {
                    wGate[l][i][j] = (random.nextDouble() - 0.5) * 0.1;
                    wUp[l][i][j]   = (random.nextDouble() - 0.5) * 0.1;
                }
            }
            for (int i = 0; i < dHidden; i++) {
                for (int j = 0; j < dModel; j++) {
                    wDown[l][i][j] = (random.nextDouble() - 0.5) * 0.1;
                }
            }
        }
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < vocabSizeLocal; j++) {
                wOut[i][j] = (random.nextDouble() - 0.5) * 0.1;
            }
        }

        // 学習時の使い回し用ワークスペース（多層対応）
        double[] invFreq = new double[vectorSize];
        for (int j = 0; j < vectorSize; j++) {
            double exponent = (double) (2 * (j / 2)) / vectorSize;
            invFreq[j] = 1.0 / Math.pow(10000.0, exponent);
        }

        double[][] inputEmbeddings = new double[maxSeqLen][vectorSize];

        // 各レイヤーの計算結果を保持できるよう 3 次元に変更
        double[][][] query = new double[numLayers][maxSeqLen][dModel];
        double[][][] key = new double[numLayers][maxSeqLen][dModel];
        double[][][] value = new double[numLayers][maxSeqLen][dModel];
        double[][][] attentionScores = new double[numLayers][maxSeqLen][maxSeqLen];
        double[][][] attentionWeights = new double[numLayers][maxSeqLen][maxSeqLen];
        double[][][] attentionOutput = new double[numLayers][maxSeqLen][dModel];
        double[][][] ffnOutput = new double[numLayers][maxSeqLen][dModel];

        double[][][] dFfnOutput = new double[numLayers][maxSeqLen][dModel];
        double[][][] dAttentionOutput = new double[numLayers][maxSeqLen][dModel];
        double[][][] dAttentionWeights = new double[numLayers][maxSeqLen][maxSeqLen];
        double[][][] dValue = new double[numLayers][maxSeqLen][dModel];
        double[][][] dAttentionScores = new double[numLayers][maxSeqLen][maxSeqLen];
        double[][][] dQuery = new double[numLayers][maxSeqLen][dModel];
        double[][][] dKey = new double[numLayers][maxSeqLen][dModel];
        double[][] dInputEmbeddings = new double[maxSeqLen][dModel];


        System.out.println("=== 学習開始 ===");

        for (int epoch = 0; epoch < epochs; epoch++) {

            String currentText = dataset[epoch % dataset.length];
            int seqLen = currentText.length();

            int[] encoded = new int[seqLen];
            for (int i = 0; i < seqLen; i++) {
                encoded[i] = charToId.get(currentText.charAt(i));
            }

            // 勾配配列のゼロクリア
            for (double[] row : dWOut)   Arrays.fill(row, 0.0);
            for (double[] row : dWGate)  Arrays.fill(row, 0.0);
            for (double[] row : dWUp)    Arrays.fill(row, 0.0);
            for (double[] row : dWDown)  Arrays.fill(row, 0.0);
            for (double[] row : dWq)     Arrays.fill(row, 0.0);
            for (double[] row : dWk)     Arrays.fill(row, 0.0);
            for (double[] row : dWv)     Arrays.fill(row, 0.0);
            for (double[] row : dEmbeddingTable) Arrays.fill(row, 0.0);

            // 4. Embedding Lookup
            for (int i = 0; i < seqLen; i++) {
                int id = encoded[i];
                System.arraycopy(embeddingTable[id], 0, inputEmbeddings[i], 0, vectorSize);
            }

            // 5. Positional Encoding
            for (int i = 0; i < seqLen; i++) {
                for (int j = 0; j < vectorSize; j++) {
                    double angle = i * invFreq[j];
                    if (j % 2 == 0) {
                        inputEmbeddings[i][j] += Math.sin(angle);
                    } else {
                        inputEmbeddings[i][j] += Math.cos(angle);
                    }
                }
            }

            // 6. Query (Q) の作成
            IntStream.range(0, seqLen).parallel().forEach(i -> {
                for (int j = 0; j < dModel; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < dModel; k++) {
                        sum += inputEmbeddings[i][k] * wq[k][j];
                    }
                    query[i][j] = sum;
                }
            });

            // 7. Key (K) の作成
            IntStream.range(0, seqLen).parallel().forEach(i -> {
                for (int j = 0; j < dModel; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < dModel; k++) {
                        sum += inputEmbeddings[i][k] * wk[k][j];
                    }
                    key[i][j] = sum;
                }
            });

            // 8. Value (V) の作成
            IntStream.range(0, seqLen).parallel().forEach(i -> {
                for (int j = 0; j < dModel; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < dModel; k++) {
                        sum += inputEmbeddings[i][k] * wv[k][j];
                    }
                    value[i][j] = sum;
                }
            });

            // 9. Causal Mask 付き Self-Attention スコア計算
            double scale = Math.sqrt(dModel);
            for (int i = 0; i < seqLen; i++) {
                Arrays.fill(attentionScores[i], 0, seqLen, -1e9);
                for (int j = 0; j <= i; j++) {
                    double dotProduct = 0.0;
                    for (int k = 0; k < dModel; k++) {
                        dotProduct += query[i][k] * key[j][k];
                    }
                    attentionScores[i][j] = dotProduct / scale;
                }
            }

            // 10. Softmax
            IntStream.range(0, seqLen).parallel().forEach(i -> {
                double max = attentionScores[i][0];
                for (int j = 1; j < seqLen; j++) {
                    if (attentionScores[i][j] > max) {
                        max = attentionScores[i][j];
                    }
                }

                double sum = 0.0;
                double[] expRow = new double[seqLen];
                for (int j = 0; j < seqLen; j++) {
                    expRow[j] = Math.exp(attentionScores[i][j] - max);
                    sum += expRow[j];
                }

                for (int j = 0; j < seqLen; j++) {
                    attentionWeights[i][j] = expRow[j] / sum;
                }
            });

            // 11. Attention Output
            IntStream.range(0, seqLen).parallel().forEach(i -> {
                for (int j = 0; j < dModel; j++) {
                    double weightedSum = 0.0;
                    for (int k = 0; k < seqLen; k++) {
                        weightedSum += attentionWeights[i][k] * value[k][j];
                    }
                    attentionOutput[i][j] = weightedSum;
                }
            });

            // 12. SwiGLU FFN
            IntStream.range(0, seqLen).parallel().forEach(i -> {
                double[] gate = new double[dHidden];
                double[] up = new double[dHidden];
                double[] gatedValue = new double[dHidden];

                for (int j = 0; j < dHidden; j++) {
                    double sumGate = 0.0;
                    double sumUp = 0.0;

                    for (int k = 0; k < dModel; k++) {
                        double val = attentionOutput[i][k];
                        sumGate += val * wGate[k][j];
                        sumUp += val * wUp[k][j];
                    }

                    double sigmoid = 1.0 / (1.0 + Math.exp(-sumGate));
                    gate[j] = sumGate * sigmoid;
                    up[j] = sumUp;
                }

                for (int j = 0; j < dHidden; j++) {
                    gatedValue[j] = gate[j] * up[j];
                }

                for (int j = 0; j < dModel; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < dHidden; k++) {
                        sum += gatedValue[k] * wDown[k][j];
                    }
                    ffnOutput[i][j] = sum;
                }
            });

            // 13. 損失計算
            int numPredictions = seqLen - 1;
            double totalLoss = IntStream.range(0, numPredictions).parallel().mapToDouble(i -> {
                int targetId = encoded[i + 1];

                double[] logits = new double[vocabSizeLocal];
                for (int j = 0; j < vocabSizeLocal; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < dModel; k++) {
                        sum += ffnOutput[i][k] * wOut[k][j];
                    }
                    logits[j] = sum;
                }

                double maxLogit = logits[0];
                for (int j = 1; j < vocabSizeLocal; j++) {
                    if (logits[j] > maxLogit) maxLogit = logits[j];
                }

                double targetLogit = logits[targetId];
                double logSumExp = 0.0;
                for (int j = 0; j < vocabSizeLocal; j++) {
                    logSumExp += Math.exp(logits[j] - maxLogit);
                }
                double logNormalizer = maxLogit + Math.log(logSumExp);

                return -(targetLogit - logNormalizer);
            }).sum();

            double finalLoss = totalLoss / numPredictions;

            // 14. 逆伝播
            for (int i = 0; i < seqLen; i++) {
                Arrays.fill(dFfnOutput[i], 0.0);
                Arrays.fill(dAttentionOutput[i], 0.0);
                Arrays.fill(dQuery[i], 0.0);
                Arrays.fill(dKey[i], 0.0);
                Arrays.fill(dValue[i], 0.0);
                Arrays.fill(dInputEmbeddings[i], 0.0);
            }
            for (int i = 0; i < seqLen; i++) {
                Arrays.fill(dAttentionWeights[i], 0.0);
                Arrays.fill(dAttentionScores[i], 0.0);
            }

            // --- [A] 出力層の逆伝播 ---
            for (int i = 0; i < numPredictions; i++) {
                int targetId = encoded[i + 1];

                double[] logits = new double[vocabSizeLocal];
                for (int j = 0; j < vocabSizeLocal; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < dModel; k++) {
                        sum += ffnOutput[i][k] * wOut[k][j];
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

                double[] dLogits = new double[vocabSizeLocal];
                System.arraycopy(probs, 0, dLogits, 0, vocabSizeLocal);
                dLogits[targetId] -= 1.0;

                for (int k = 0; k < dModel; k++) {
                    for (int j = 0; j < vocabSizeLocal; j++) {
                        dWOut[k][j] += ffnOutput[i][k] * dLogits[j];
                    }
                    double gradSum = 0.0;
                    for (int j = 0; j < vocabSizeLocal; j++) {
                        gradSum += dLogits[j] * wOut[k][j];
                    }
                    dFfnOutput[i][k] += gradSum;
                }
            }

            // --- [B] FFN層の逆伝播 ---
            for (int i = 0; i < seqLen; i++) {
                double[] gate = new double[dHidden];
                double[] up = new double[dHidden];
                double[] sigmoidArr = new double[dHidden];
                double[] sumGateArr = new double[dHidden];

                for (int j = 0; j < dHidden; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < dModel; k++) {
                        sum += attentionOutput[i][k] * wGate[k][j];
                    }
                    sumGateArr[j] = sum;
                    double sigmoid = 1.0 / (1.0 + Math.exp(-sum));
                    sigmoidArr[j] = sigmoid;
                    gate[j] = sum * sigmoid;
                }

                for (int j = 0; j < dHidden; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < dModel; k++) {
                        sum += attentionOutput[i][k] * wUp[k][j];
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
                        sum += dFfnOutput[i][j] * wDown[k][j];
                        dWDown[k][j] += gatedValue[k] * dFfnOutput[i][j];
                    }
                    dGatedValue[k] = sum;
                }

                double[] dGate = new double[dHidden];
                double[] dUp = new double[dHidden];
                for (int k = 0; k < dHidden; k++) {
                    dGate[k] = dGatedValue[k] * up[k];
                    dUp[k] = dGatedValue[k] * gate[k];
                }

                for (int j = 0; j < dHidden; j++) {
                    double sum = sumGateArr[j];
                    double s = sigmoidArr[j];
                    double dSilu = s + (sum * s) * (1.0 - s);
                    double dSumGate = dGate[j] * dSilu;

                    for (int k = 0; k < dModel; k++) {
                        dWGate[k][j] += attentionOutput[i][k] * dSumGate;
                        dWUp[k][j]   += attentionOutput[i][k] * dUp[j];
                    }
                }
            }

            // パラメータ更新 (出力・FFN)
            for (int i = 0; i < dModel; i++) {
                for (int j = 0; j < vocabSizeLocal; j++) {
                    wOut[i][j] -= learningRate * dWOut[i][j];
                }
            }
            for (int i = 0; i < dModel; i++) {
                for (int j = 0; j < dHidden; j++) {
                    wGate[i][j] -= learningRate * dWGate[i][j];
                    wUp[i][j]   -= learningRate * dWUp[i][j];
                }
            }
            for (int i = 0; i < dHidden; i++) {
                for (int j = 0; j < dModel; j++) {
                    wDown[i][j] -= learningRate * dWDown[i][j];
                }
            }

            // --- [C] Attention層の逆伝播 ---
            for (int i = 0; i < seqLen; i++) {
                double[] dGatedValue = new double[dHidden];
                for (int k = 0; k < dHidden; k++) {
                    double sum = 0.0;
                    for (int j = 0; j < dModel; j++) {
                        sum += dFfnOutput[i][j] * wDown[k][j];
                    }
                    dGatedValue[k] = sum;
                }

                double[] gate = new double[dHidden];
                double[] up = new double[dHidden];
                double[] sigmoidArr = new double[dHidden];
                double[] sumGateArr = new double[dHidden];

                for (int j = 0; j < dHidden; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < dModel; k++) {
                        sum += attentionOutput[i][k] * wGate[k][j];
                    }
                    sumGateArr[j] = sum;
                    double sigmoid = 1.0 / (1.0 + Math.exp(-sum));
                    sigmoidArr[j] = sigmoid;
                    gate[j] = sum * sigmoid;
                }
                for (int j = 0; j < dHidden; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < dModel; k++) {
                        sum += attentionOutput[i][k] * wUp[k][j];
                    }
                    up[j] = sum;
                }

                double[] dGate = new double[dHidden];
                double[] dUp = new double[dHidden];
                for (int k = 0; k < dHidden; k++) {
                    dGate[k] = dGatedValue[k] * up[k];
                    dUp[k] = dGatedValue[k] * gate[k];
                }

                for (int j = 0; j < dHidden; j++) {
                    double sum = sumGateArr[j];
                    double s = sigmoidArr[j];
                    double dSilu = s + (sum * s) * (1.0 - s);
                    double dSumGate = dGate[j] * dSilu;

                    for (int k = 0; k < dModel; k++) {
                        dAttentionOutput[i][k] += dSumGate * wGate[k][j] + dUp[j] * wUp[k][j];
                    }
                }
            }

            for (int i = 0; i < seqLen; i++) {
                for (int j = 0; j < dModel; j++) {
                    double gradOut = dAttentionOutput[i][j];
                    for (int k = 0; k < seqLen; k++) {
                        dAttentionWeights[i][k] += gradOut * value[k][j];
                        dValue[k][j] += gradOut * attentionWeights[i][k];
                    }
                }
            }

            for (int i = 0; i < seqLen; i++) {
                double dotSum = 0.0;
                for (int k = 0; k < seqLen; k++) {
                    dotSum += dAttentionWeights[i][k] * attentionWeights[i][k];
                }
                for (int j = 0; j < seqLen; j++) {
                    double y_j = attentionWeights[i][j];
                    dAttentionScores[i][j] = y_j * (dAttentionWeights[i][j] - dotSum);
                }
            }

            for (int i = 0; i < seqLen; i++) {
                for (int j = 0; j < seqLen; j++) {
                    if (j > i) {
                        dAttentionScores[i][j] = 0.0;
                    }
                }
            }

            scale = Math.sqrt(dModel);
            for (int i = 0; i < seqLen; i++) {
                for (int j = 0; j < seqLen; j++) {
                    double dScore = dAttentionScores[i][j] / scale;
                    for (int k = 0; k < dModel; k++) {
                        dQuery[i][k] += dScore * key[j][k];
                        dKey[j][k]   += dScore * query[i][k];
                    }
                }
            }

            for (int i = 0; i < seqLen; i++) {
                for (int j = 0; j < dModel; j++) {
                    double gQ = dQuery[i][j];
                    double gK = dKey[i][j];
                    double gV = dValue[i][j];

                    for (int k = 0; k < dModel; k++) {
                        dWq[k][j] += inputEmbeddings[i][k] * gQ;
                        dWk[k][j] += inputEmbeddings[i][k] * gK;
                        dWv[k][j] += inputEmbeddings[i][k] * gV;

                        dInputEmbeddings[i][k] += gQ * wq[k][j] + gK * wk[k][j] + gV * wv[k][j];
                    }
                }
            }

            for (int i = 0; i < dModel; i++) {
                for (int j = 0; j < dModel; j++) {
                    wq[i][j] -= learningRate * dWq[i][j];
                    wk[i][j] -= learningRate * dWk[i][j];
                    wv[i][j] -= learningRate * dWv[i][j];
                }
            }

            for (int i = 0; i < seqLen; i++) {
                int id = encoded[i];
                for (int j = 0; j < vectorSize; j++) {
                    dEmbeddingTable[id][j] += dInputEmbeddings[i][j];
                }
            }

            for (int i = 0; i < vocabSize; i++) {
                for (int j = 0; j < vectorSize; j++) {
                    embeddingTable[i][j] -= learningRate * dEmbeddingTable[i][j];
                }
            }

            if (epoch == 0 || (epoch + 1) % 500 == 0 || epoch == epochs - 1) {
                System.out.println("Epoch [" + (epoch + 1) + "/" + epochs + "] - Loss: " + finalLoss);
            }
        }

        System.out.println("\n=== 対話・文字生成テスト ===");
        Scanner scanner = new Scanner(System.in);

        while (true) {
            System.out.print("AIへの入力文字をどうぞ: ");
            String userInput = scanner.nextLine();

            int[] genEncoded = new int[userInput.length()];
            for (int i = 0; i < userInput.length(); i++) {
                char c = userInput.charAt(i);
                genEncoded[i] = charToId.getOrDefault(c, 0);
            }

            System.out.print("入力: " + userInput + "  生成結果: " + userInput);

            for (int step = 0; step < 30; step++) {
                int genSeqLen = genEncoded.length;

                for (int i = 0; i < genSeqLen; i++) {
                    int id = genEncoded[i];
                    System.arraycopy(embeddingTable[id], 0, inputEmbeddings[i], 0, vectorSize);
                }

                for (int i = 0; i < genSeqLen; i++) {
                    for (int j = 0; j < vectorSize; j++) {
                        double angle = i / Math.pow(10000.0, (double) (2 * (j / 2)) / vectorSize);
                        if (j % 2 == 0) {
                            inputEmbeddings[i][j] += Math.sin(angle);
                        } else {
                            inputEmbeddings[i][j] += Math.cos(angle);
                        }
                    }
                }

                for (int i = 0; i < genSeqLen; i++) {
                    for (int j = 0; j < dModel; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < dModel; k++) {
                            sum += inputEmbeddings[i][k] * wq[k][j];
                        }
                        query[i][j] = sum;
                    }
                }

                for (int i = 0; i < genSeqLen; i++) {
                    for (int j = 0; j < dModel; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < dModel; k++) {
                            sum += inputEmbeddings[i][k] * wk[k][j];
                        }
                        key[i][j] = sum;
                    }
                }

                for (int i = 0; i < genSeqLen; i++) {
                    for (int j = 0; j < dModel; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < dModel; k++) {
                            sum += inputEmbeddings[i][k] * wv[k][j];
                        }
                        value[i][j] = sum;
                    }
                }

                double scale = Math.sqrt(dModel);
                for (int i = 0; i < genSeqLen; i++) {
                    for (int j = 0; j < genSeqLen; j++) {
                        if (j > i) {
                            attentionScores[i][j] = -1e9;
                        } else {
                            double dotProduct = 0.0;
                            for (int k = 0; k < dModel; k++) {
                                dotProduct += query[i][k] * key[j][k];
                            }
                            attentionScores[i][j] = dotProduct / scale;
                        }
                    }
                }

                for (int i = 0; i < genSeqLen; i++) {
                    double max = attentionScores[i][0];
                    for (int j = 1; j < genSeqLen; j++) {
                        if (attentionScores[i][j] > max) {
                            max = attentionScores[i][j];
                        }
                    }

                    double sum = 0.0;
                    double[] expRow = new double[genSeqLen];
                    for (int j = 0; j < genSeqLen; j++) {
                        expRow[j] = Math.exp(attentionScores[i][j] - max);
                        sum += expRow[j];
                    }

                    for (int j = 0; j < genSeqLen; j++) {
                        attentionWeights[i][j] = expRow[j] / sum;
                    }
                }

                for (int i = 0; i < genSeqLen; i++) {
                    for (int j = 0; j < dModel; j++) {
                        double weightedSum = 0.0;
                        for (int k = 0; k < genSeqLen; k++) {
                            weightedSum += attentionWeights[i][k] * value[k][j];
                        }
                        attentionOutput[i][j] = weightedSum;
                    }
                }

                for (int i = 0; i < genSeqLen; i++) {
                    double[] gate = new double[dHidden];
                    for (int j = 0; j < dHidden; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < dModel; k++) {
                            sum += attentionOutput[i][k] * wGate[k][j];
                        }
                        double sigmoid = 1.0 / (1.0 + Math.exp(-sum));
                        gate[j] = sum * sigmoid;
                    }

                    double[] up = new double[dHidden];
                    for (int j = 0; j < dHidden; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < dModel; k++) {
                            sum += attentionOutput[i][k] * wUp[k][j];
                        }
                        up[j] = sum;
                    }

                    double[] gatedValue = new double[dHidden];
                    for (int j = 0; j < dHidden; j++) {
                        gatedValue[j] = gate[j] * up[j];
                    }

                    for (int j = 0; j < dModel; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < dHidden; k++) {
                            sum += gatedValue[k] * wDown[k][j];
                        }
                        ffnOutput[i][j] = sum;
                    }
                }

                int lastIdx = genSeqLen - 1;
                double[] logits = new double[vocabSizeLocal];
                for (int j = 0; j < vocabSizeLocal; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < dModel; k++) {
                        sum += ffnOutput[lastIdx][k] * wOut[k][j];
                    }
                    logits[j] = sum;
                }

                int bestNextId = 0;
                double maxLogitVal = logits[0];
                for (int j = 1; j < vocabSizeLocal; j++) {
                    if (logits[j] > maxLogitVal) {
                        maxLogitVal = logits[j];
                        bestNextId = j;
                    }
                }

                char predictedChar = idToChar.get(bestNextId);
                System.out.print(predictedChar);

                int[] nextGenEncoded = new int[genSeqLen + 1];
                System.arraycopy(genEncoded, 0, nextGenEncoded, 0, genSeqLen);
                nextGenEncoded[genSeqLen] = bestNextId;
                genEncoded = nextGenEncoded;
            }
            System.out.println();
        }
    }
}