package com.mc1510ty.TekitouLLM;

import java.util.*;

public class Main {

    static void main() {

        long seed = 12345L;
        Random random = new Random(seed);

        // 1. 複数の会話パターン（データセット）を用意する
        String[] dataset = {
                "U:こんにちは>A:こんにちは、良い天気ですね。■",
                "U:おはよう>A:おはようございます、今朝は何時に起きましたか？■",
                "U:こんばんは>A:こんばんは、月がきれいですね。■",
                "U:天気は？>A:今日は雨予想ですね。■",
                "U:眠いな>A:今朝はよく眠れましたか？時間があればお昼寝をしたりするのも手です。■",
                "U:明日も学校かぁ>A:毎日毎日、同じ日々の繰り返しで大変ですよね。睡眠時間をしっかり取るとまだましになるはずです。■",
                "U:LLMって何？>A:LLMとは、簡単に言うと現代の会話できるAIのことです。■",
                "U:Java言語の名前の由来は何？>A:Java(プログラミング言語)の名前の由来は、開発チームが好んで飲んでいたインドネシア・ジャワ島のコーヒーに由来しています。Javaの歴史についてもっと知りたい場合は、他の質問もどんどん言ってください!■",
                "U:Javaの最新バージョンは？>A:Java(プログラミング言語)の最新LTSバージョンは、Java25の、'25.0.4.1'です。Java25では、クラス定義の簡略化や、IO.printlnが使用可能になるなど、より便利になりました。■",
                "U:ls>A:lsは、LinuxやMacOSなどのターミナルで、ディレクトリやファイルの一覧を表示する基本コマンドです。■",
                "U:桃太郎の桃が流れる音>A:'どんぶらこ'です。■",
                "U:OpenGLの代替は？>A:主に'Vulkan'が挙げられます。Windows専用であれば、'DirectX'なども可能です。VulkanやDirectXについて質問がある場合は、なんでも言ってください！■",
                "U:こんにちは>A:こんにちは！今日は何をしますか？■",
                "U:こんばんは>A:こんばんは。今日は何がありましたか?■",
                "U:Windowsの最新バージョン>A:Microsoft Windowsの2026年10月2日時点での最新バージョンは、2021年10月5日にリリースされた'Windows 11'です。■",
                "U:BVE>A:'BVE'は、mackoy氏が主に個人で開発している、3DCGを用いたトレインシミュレーターのことです。最新バージョンは2020年9月23日にリリースされた、'BVE6'です。公式サイトのURLは、'https://bvets.net/'となっています。"
        };


        // 学習設定
        int epochs = 5000;
        double learningRate = 0.0005;
        int numLayers = 8;

        int maxSeqLen = 2048;


        int vectorSize = 32;

        // 2. BPEトークナイザーの初期化と学習
        SimpleTokenizer tokenizer = new SimpleTokenizer();
        // 例として、目標語彙サイズを320、最小頻度を2に設定して学習
        tokenizer.train(dataset, 320, 2);

        System.out.println("\n作成されたサブワード:");
        tokenizer.tokenToId.entrySet().stream().forEach(entry -> {
            System.out.println("  [" + entry.getValue() + "] " + entry.getKey());
        });


        int vocabSize = tokenizer.getVocabSize();
        int dModel = vectorSize;
        int dHidden = dModel * 2; //隠れ層は2倍
        int vocabSizeLocal = vocabSize;

        // 3. 埋め込みテーブルの初期化
        double[][] embeddingTable = new double[vocabSize][vectorSize];
        for (int i = 0; i < vocabSize; i++) {
            for (int j = 0; j < vectorSize; j++) {
                embeddingTable[i][j] = (random.nextDouble() - 0.5);
            }
        }

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

        // 重みのランダム初期化
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

        double[] invFreq = new double[vectorSize];
        for (int j = 0; j < vectorSize; j++) {
            double exponent = (double) (2 * (j / 2)) / vectorSize;
            invFreq[j] = 1.0 / Math.pow(10000.0, exponent);
        }

        double[][] inputEmbeddings = new double[maxSeqLen][vectorSize];

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

            // BPEでエンコードしてトークンIDのリストを取得
            List<Integer> encodedList = tokenizer.encode(currentText);
            int seqLen = encodedList.size();

            int[] encoded = new int[seqLen];
            for (int i = 0; i < seqLen; i++) {
                encoded[i] = encodedList.get(i);
            }

            // 勾配配列のゼロクリア
            for (double[] row : dWOut)   Arrays.fill(row, 0.0);
            for (double[] row : dEmbeddingTable) Arrays.fill(row, 0.0);

            for (double[][] matrix : dWGate)  for (double[] row : matrix) Arrays.fill(row, 0.0);
            for (double[][] matrix : dWUp)    for (double[] row : matrix) Arrays.fill(row, 0.0);
            for (double[][] matrix : dWDown)  for (double[] row : matrix) Arrays.fill(row, 0.0);
            for (double[][] matrix : dWq)     for (double[] row : matrix) Arrays.fill(row, 0.0);
            for (double[][] matrix : dWk)     for (double[] row : matrix) Arrays.fill(row, 0.0);
            for (double[][] matrix : dWv)     for (double[] row : matrix) Arrays.fill(row, 0.0);

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

            // --- 順伝播 ---
            for (int l = 0; l < numLayers; l++) {
                final int layer = l;
                double[][] currentLayerInput = (layer == 0) ? inputEmbeddings : ffnOutput[layer - 1];

                // 6. Query (Q)
                for (int i = 0; i < seqLen; i++) {
                    for (int j = 0; j < dModel; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < dModel; k++) {
                            sum += currentLayerInput[i][k] * wq[layer][k][j];
                        }
                        query[layer][i][j] = sum;
                    }
                }

                // 7. Key (K)
                for (int i = 0; i < seqLen; i++) {
                    for (int j = 0; j < dModel; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < dModel; k++) {
                            sum += currentLayerInput[i][k] * wk[layer][k][j];
                        }
                        key[layer][i][j] = sum;
                    }
                }

                // 8. Value (V)
                for (int i = 0; i < seqLen; i++) {
                    for (int j = 0; j < dModel; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < dModel; k++) {
                            sum += currentLayerInput[i][k] * wv[layer][k][j];
                        }
                        value[layer][i][j] = sum;
                    }
                }

                // 9. Causal Mask 付き Attention スコア
                double scale = Math.sqrt(dModel);
                for (int i = 0; i < seqLen; i++) {
                    Arrays.fill(attentionScores[layer][i], 0, seqLen, -1e9);
                    for (int j = 0; j <= i; j++) {
                        double dotProduct = 0.0;
                        for (int k = 0; k < dModel; k++) {
                            dotProduct += query[layer][i][k] * key[layer][j][k];
                        }
                        attentionScores[layer][i][j] = dotProduct / scale;
                    }
                }

                // 10. Softmax
                for (int i = 0; i < seqLen; i++) {
                    double max = attentionScores[layer][i][0];
                    for (int j = 1; j < seqLen; j++) {
                        if (attentionScores[layer][i][j] > max) {
                            max = attentionScores[layer][i][j];
                        }
                    }

                    double sum = 0.0;
                    double[] expRow = new double[seqLen];
                    for (int j = 0; j < seqLen; j++) {
                        expRow[j] = Math.exp(attentionScores[layer][i][j] - max);
                        sum += expRow[j];
                    }

                    for (int j = 0; j < seqLen; j++) {
                        attentionWeights[layer][i][j] = expRow[j] / sum;
                    }
                }

                // 11. Attention Output (★ 残差接続: ＋ currentLayerInput)
                for (int i = 0; i < seqLen; i++) {
                    for (int j = 0; j < dModel; j++) {
                        double weightedSum = 0.0;
                        for (int k = 0; k < seqLen; k++) {
                            weightedSum += attentionWeights[layer][i][k] * value[layer][k][j];
                        }
                        attentionOutput[layer][i][j] = weightedSum + currentLayerInput[i][j];
                    }
                }

                // 12. SwiGLU FFN (★ 残差接続: ＋ attentionOutput)
                for (int i = 0; i < seqLen; i++) {
                    double[] gate = new double[dHidden];
                    double[] up = new double[dHidden];
                    double[] gatedValue = new double[dHidden];

                    for (int j = 0; j < dHidden; j++) {
                        double sumGate = 0.0;
                        double sumUp = 0.0;

                        for (int k = 0; k < dModel; k++) {
                            double val = attentionOutput[layer][i][k];
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

                    for (int j = 0; j < dModel; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < dHidden; k++) {
                            sum += gatedValue[k] * wDown[layer][k][j];
                        }
                        ffnOutput[layer][i][j] = sum + attentionOutput[layer][i][j];
                    }
                }
            }

            // 13. 損失計算
            int numPredictions = seqLen - 1;
            double totalLoss = 0.0;
            for (int i = 0; i < numPredictions; i++) {
                int targetId = encoded[i + 1];

                double[] logits = new double[vocabSizeLocal];
                for (int j = 0; j < vocabSizeLocal; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < dModel; k++) {
                        sum += ffnOutput[numLayers - 1][i][k] * wOut[k][j];
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

                totalLoss += -(targetLogit - logNormalizer);
            }

            double finalLoss = totalLoss / numPredictions;

            // 14. 逆伝播ワークスペース初期化
            for (int l = 0; l < numLayers; l++) {
                for (int i = 0; i < seqLen; i++) {
                    Arrays.fill(dFfnOutput[l][i], 0.0);
                    Arrays.fill(dAttentionOutput[l][i], 0.0);
                    Arrays.fill(dQuery[l][i], 0.0);
                    Arrays.fill(dKey[l][i], 0.0);
                    Arrays.fill(dValue[l][i], 0.0);
                }
                for (int i = 0; i < seqLen; i++) {
                    Arrays.fill(dAttentionWeights[l][i], 0.0);
                    Arrays.fill(dAttentionScores[l][i], 0.0);
                }
            }
            for (int i = 0; i < seqLen; i++) {
                Arrays.fill(dInputEmbeddings[i], 0.0);
            }

            // --- [A] 出力層の逆伝播 ---
            for (int i = 0; i < numPredictions; i++) {
                int targetId = encoded[i + 1];

                double[] logits = new double[vocabSizeLocal];
                for (int j = 0; j < vocabSizeLocal; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < dModel; k++) {
                        sum += ffnOutput[numLayers - 1][i][k] * wOut[k][j];
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
                        dWOut[k][j] += ffnOutput[numLayers - 1][i][k] * dLogits[j];
                    }
                    double gradSum = 0.0;
                    for (int j = 0; j < vocabSizeLocal; j++) {
                        gradSum += dLogits[j] * wOut[k][j];
                    }
                    dFfnOutput[numLayers - 1][i][k] += gradSum;
                }
            }

            // --- 逆伝播ループ (★ 残差接続による勾配のバイパスを追加) ---
            for (int l = numLayers - 1; l >= 0; l--) {
                double[][] currentLayerInput = (l == 0) ? inputEmbeddings : ffnOutput[l - 1];

                // FFNの残差接続による勾配のバイパス (dFfnOutput の勾配がそのまま AttentionOutput にも流れる)
                for (int i = 0; i < seqLen; i++) {
                    for (int k = 0; k < dModel; k++) {
                        dAttentionOutput[l][i][k] += dFfnOutput[l][i][k];
                    }
                }

                // --- [B] FFN層の逆伝播（パラメータの勾配と入力への勾配をここでまとめて計算） ---
                for (int i = 0; i < seqLen; i++) {
                    double[] sumGateArr = new double[dHidden];
                    double[] sigmoidArr = new double[dHidden];
                    double[] gate = new double[dHidden];
                    double[] up = new double[dHidden];

                    for (int j = 0; j < dHidden; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < dModel; k++) {
                            sum += attentionOutput[l][i][k] * wGate[l][k][j];
                        }
                        sumGateArr[j] = sum;
                        double sigmoid = 1.0 / (1.0 + Math.exp(-sum));
                        sigmoidArr[j] = sigmoid;
                        gate[j] = sum * sigmoid;
                    }

                    for (int j = 0; j < dHidden; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < dModel; k++) {
                            sum += attentionOutput[l][i][k] * wUp[l][k][j];
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
                            sum += dFfnOutput[l][i][j] * wDown[l][k][j];
                            dWDown[l][k][j] += gatedValue[k] * dFfnOutput[l][i][j];
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
                            // パラメータの勾配
                            dWGate[l][k][j] += attentionOutput[l][i][k] * dSumGate;
                            dWUp[l][k][j]   += attentionOutput[l][i][k] * dUp_j;

                            // 入力（AttentionOutput）へ流す勾配
                            dAttentionOutput[l][i][k] += dSumGate * wGate[l][k][j] + dUp_j * wUp[l][k][j];
                        }
                    }
                }


                for (int i = 0; i < seqLen; i++) {
                    for (int j = 0; j < dModel; j++) {
                        double gradOut = dAttentionOutput[l][i][j];
                        for (int k = 0; k < seqLen; k++) {
                            dAttentionWeights[l][i][k] += gradOut * value[l][k][j];
                            dValue[l][k][j] += gradOut * attentionWeights[l][i][k];
                        }
                    }
                }

                for (int i = 0; i < seqLen; i++) {
                    double dotSum = 0.0;
                    for (int k = 0; k < seqLen; k++) {
                        dotSum += dAttentionWeights[l][i][k] * attentionWeights[l][i][k];
                    }
                    for (int j = 0; j < seqLen; j++) {
                        double y_j = attentionWeights[l][i][j];
                        dAttentionScores[l][i][j] = y_j * (dAttentionWeights[l][i][j] - dotSum);
                    }
                }

                for (int i = 0; i < seqLen; i++) {
                    for (int j = 0; j < seqLen; j++) {
                        if (j > i) {
                            dAttentionScores[l][i][j] = 0.0;
                        }
                    }
                }

                double scale = Math.sqrt(dModel);
                for (int i = 0; i < seqLen; i++) {
                    for (int j = 0; j < seqLen; j++) {
                        double dScore = dAttentionScores[l][i][j] / scale;
                        for (int k = 0; k < dModel; k++) {
                            dQuery[l][i][k] += dScore * key[l][j][k];
                            dKey[l][j][k]   += dScore * query[l][i][k];
                        }
                    }
                }

                // Attentionの残差接続による勾配のバイパス (dAttentionOutput の勾配がそのまま入力にも流れる)
                for (int i = 0; i < seqLen; i++) {
                    for (int k = 0; k < dModel; k++) {
                        double gradOut = dAttentionOutput[l][i][k];
                        if (l > 0) {
                            dFfnOutput[l - 1][i][k] += gradOut;
                        } else {
                            dInputEmbeddings[i][k] += gradOut;
                        }
                    }
                }

                for (int i = 0; i < seqLen; i++) {
                    for (int j = 0; j < dModel; j++) {
                        double gQ = dQuery[l][i][j];
                        double gV = dValue[l][i][j];

                        for (int k = 0; k < dModel; k++) {
                            dWq[l][k][j] += currentLayerInput[i][k] * gQ;
                            dWv[l][k][j] += currentLayerInput[i][k] * gV;

                            double gradInputVal = gQ * wq[l][k][j] + gV * wv[l][k][j];

                            if (l > 0) {
                                dFfnOutput[l - 1][i][k] += gradInputVal;
                            } else {
                                dInputEmbeddings[i][k] += gradInputVal;
                            }
                        }
                    }
                }

                for (int j = 0; j < seqLen; j++) {
                    for (int k = 0; k < dModel; k++) {
                        double gK = dKey[l][j][k];
                        for (int m = 0; m < dModel; m++) {
                            dWk[l][m][k] += currentLayerInput[j][m] * gK;

                            double gradInputValK = gK * wk[l][m][k];
                            if (l > 0) {
                                dFfnOutput[l - 1][j][m] += gradInputValK;
                            } else {
                                dInputEmbeddings[j][m] += gradInputValK;
                            }
                        }
                    }
                }
            }

            // パラメータ更新
            for (int i = 0; i < dModel; i++) {
                for (int j = 0; j < vocabSizeLocal; j++) {
                    wOut[i][j] -= learningRate * dWOut[i][j];
                }
            }
            for (int l = 0; l < numLayers; l++) {
                for (int i = 0; i < dModel; i++) {
                    for (int j = 0; j < dHidden; j++) {
                        wGate[l][i][j] -= learningRate * dWGate[l][i][j];
                        wUp[l][i][j]   -= learningRate * dWUp[l][i][j];
                    }
                }
                for (int i = 0; i < dHidden; i++) {
                    for (int j = 0; j < dModel; j++) {
                        wDown[l][i][j] -= learningRate * dWDown[l][i][j];
                    }
                }
                for (int i = 0; i < dModel; i++) {
                    for (int j = 0; j < dModel; j++) {
                        wq[l][i][j] -= learningRate * dWq[l][i][j];
                        wk[l][i][j] -= learningRate * dWk[l][i][j];
                        wv[l][i][j] -= learningRate * dWv[l][i][j];
                    }
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

        // 推論テスト部分はそのまま
        System.out.println("\n=== 対話・文字生成テスト ===");
        Scanner scanner = new Scanner(System.in);

        while (true) {
            System.out.print("AIへの入力文字をどうぞ: ");
            String userInput = scanner.nextLine();

            // AIへのプロンプト（質問）の形を作る
            String prompt = "U:" + userInput + ">A:";

            // BPEでプロンプトをエンコード
            List<Integer> promptEncodedList = tokenizer.encode(prompt);
            int[] genEncoded = new int[promptEncodedList.size()];
            for (int i = 0; i < promptEncodedList.size(); i++) {
                genEncoded[i] = promptEncodedList.get(i);
            }

            System.out.print("入力: " + userInput + "  生成結果: " + prompt);


            for (int step = 0; step < 100; step++) {
                int genSeqLen = genEncoded.length;

                for (int i = 0; i < genSeqLen; i++) {
                    int id = genEncoded[i];
                    System.arraycopy(embeddingTable[id], 0, inputEmbeddings[i], 0, vectorSize);
                }

                for (int i = 0; i < genSeqLen; i++) {
                    for (int j = 0; j < vectorSize; j++) {
                        double angle = i * invFreq[j]; // ← 事前に計算した invFreq を使うように変更
                        if (j % 2 == 0) {
                            inputEmbeddings[i][j] += Math.sin(angle);
                        } else {
                            inputEmbeddings[i][j] += Math.cos(angle);
                        }
                    }
                }

                for (int l = 0; l < numLayers; l++) {
                    double[][] currentLayerInput = (l == 0) ? inputEmbeddings : ffnOutput[l - 1];

                    for (int i = 0; i < genSeqLen; i++) {
                        for (int j = 0; j < dModel; j++) {
                            double sumQ = 0.0, sumK = 0.0, sumV = 0.0;
                            for (int k = 0; k < dModel; k++) {
                                sumQ += currentLayerInput[i][k] * wq[l][k][j];
                                sumK += currentLayerInput[i][k] * wk[l][k][j];
                                sumV += currentLayerInput[i][k] * wv[l][k][j];
                            }
                            query[l][i][j] = sumQ;
                            key[l][i][j] = sumK;
                            value[l][i][j] = sumV;
                        }
                    }

                    double scale = Math.sqrt(dModel);
                    for (int i = 0; i < genSeqLen; i++) {
                        for (int j = 0; j < genSeqLen; j++) {
                            if (j > i) {
                                attentionScores[l][i][j] = -1e9;
                            } else {
                                double dotProduct = 0.0;
                                for (int k = 0; k < dModel; k++) {
                                    dotProduct += query[l][i][k] * key[l][j][k];
                                }
                                attentionScores[l][i][j] = dotProduct / scale;
                            }
                        }
                    }

                    for (int i = 0; i < genSeqLen; i++) {
                        double max = attentionScores[l][i][0];
                        for (int j = 1; j < genSeqLen; j++) {
                            if (attentionScores[l][i][j] > max) {
                                max = attentionScores[l][i][j];
                            }
                        }

                        double sum = 0.0;
                        double[] expRow = new double[genSeqLen];
                        for (int j = 0; j < genSeqLen; j++) {
                            expRow[j] = Math.exp(attentionScores[l][i][j] - max);
                            sum += expRow[j];
                        }

                        for (int j = 0; j < genSeqLen; j++) {
                            attentionWeights[l][i][j] = expRow[j] / sum;
                        }
                    }

                    for (int i = 0; i < genSeqLen; i++) {
                        for (int j = 0; j < dModel; j++) {
                            double weightedSum = 0.0;
                            for (int k = 0; k < genSeqLen; k++) {
                                weightedSum += attentionWeights[l][i][k] * value[l][k][j];
                            }
                            attentionOutput[l][i][j] = weightedSum + currentLayerInput[i][j];
                        }
                    }

                    for (int i = 0; i < genSeqLen; i++) {
                        double[] gate = new double[dHidden];
                        double[] up = new double[dHidden];
                        for (int j = 0; j < dHidden; j++) {
                            double sumGate = 0.0, sumUp = 0.0;
                            for (int k = 0; k < dModel; k++) {
                                double val = attentionOutput[l][i][k];
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
                            ffnOutput[l][i][j] = sum + attentionOutput[l][i][j];
                        }
                    }
                }

                int lastIdx = genSeqLen - 1;
                double[] logits = new double[vocabSizeLocal];
                for (int j = 0; j < vocabSizeLocal; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < dModel; k++) {
                        sum += ffnOutput[numLayers - 1][lastIdx][k] * wOut[k][j];
                    }
                    logits[j] = sum;
                }

// 温度パラメータ（小さいほど、確率が高いものに厳しくなる。例: 0.7）
                double temperature = 0.7;

                double[] probs = new double[vocabSizeLocal];
                double sumExp = 0.0;
                double maxLogit = logits[0];
                for (int j = 1; j < vocabSizeLocal; j++) {
                    if (logits[j] > maxLogit) {
                        maxLogit = logits[j];
                    }
                }

// logitsをtemperatureで割ることで、確率のメリハリを強める
                for (int j = 0; j < vocabSizeLocal; j++) {
                    probs[j] = Math.exp((logits[j] - maxLogit) / temperature);
                    sumExp += probs[j];
                }

                for (int j = 0; j < vocabSizeLocal; j++) {
                    probs[j] /= sumExp;
                }

// 2. 確率のルーレットでランダムに次の文字を選ぶ
                double r = random.nextDouble(); // 0.0 から 1.0 未満のランダムな値
                double cumulative = 0.0;
                int bestNextId = 0;

                for (int j = 0; j < vocabSizeLocal; j++) {
                    cumulative += probs[j];
                    if (r <= cumulative) {
                        bestNextId = j;
                        break;
                    }
                }

                String predictedToken = tokenizer.decodeToken(bestNextId);

                // 終了トークン（■）の判定
                if (predictedToken.equals("■")) {
                    break;
                }

                System.out.print(predictedToken);

                int[] nextGenEncoded = new int[genSeqLen + 1];
                System.arraycopy(genEncoded, 0, nextGenEncoded, 0, genSeqLen);
                nextGenEncoded[genSeqLen] = bestNextId;
                genEncoded = nextGenEncoded;
            }
            System.out.println();
        }
    }
}