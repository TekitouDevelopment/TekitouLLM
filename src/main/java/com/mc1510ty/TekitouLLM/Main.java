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

        // ステップ1: 【並列処理】dataset全体の文字スキャンと重複排除を全コアで爆速で行う！
        java.util.List<Character> uniqueChars = java.util.Arrays.stream(dataset)
                .parallel()
                .flatMapToInt(String::chars)
                .mapToObj(c -> (char) c)
                .distinct() // 重複を消す
                .sorted()   // ★超重要：IDが毎回バラバラにならないように順番を固定する
                .collect(java.util.stream.Collectors.toList());

        // ステップ2: 【順番処理】集まった文字たちに 0, 1, 2... と綺麗にIDを振る
        int nextId = 0;
        for (char c : uniqueChars) {
            charToId.put(c, nextId);
            idToChar.put(nextId, c);
            nextId++;
        }

        int vocabSize = charToId.size(); // 辞書にある文字の種類数
        int vectorSize = 64;             // 空間の次元数（特徴の数）
        int dModel = vectorSize;

        // 3. 16次元の埋め込みテーブルを初期化
        double[][] embeddingTable = new double[vocabSize][vectorSize];
        for (int i = 0; i < vocabSize; i++) {
            for (int j = 0; j < vectorSize; j++) {
                embeddingTable[i][j] = (random.nextDouble() - 0.5);
            }
        }

        // ==========================================
        // ★ここから学習ループをスタート！
        // ==========================================
        int epochs = 100000;
        double learningRate = 0.0005;

        // 勾配配列の事前確保（ループの外で1度だけnewする）
        double[][] dWOut   = new double[vectorSize /*dModel*/][vocabSize];
        double[][] dWGate  = new double[vectorSize][vectorSize * 2];
        double[][] dWUp    = new double[vectorSize][vectorSize * 2];
        double[][] dWDown  = new double[vectorSize * 2][vectorSize];
        double[][] dWq     = new double[vectorSize][vectorSize];
        double[][] dWk     = new double[vectorSize][vectorSize];
        double[][] dWv     = new double[vectorSize][vectorSize];
        double[][] dEmbeddingTable = new double[vocabSize][vectorSize];



        //6. の初期化
        double[][] wq = new double[dModel][dModel];

        // Wq（Q用の重み行列）をランダム初期化
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < dModel; j++) {
                wq[i][j] = (random.nextDouble() - 0.5) * 0.1;
            }
        }

        //7. の初期化
        double[][] wk = new double[dModel][dModel];

        // Wk（K用の重み行列）をランダム初期化
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < dModel; j++) {
                wk[i][j] = (random.nextDouble() - 0.5) * 0.1;
            }
        }

        //8. の初期化
        double[][] wv = new double[dModel][dModel];

        // Wv（V用の重み行列）をランダム初期化
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < dModel; j++) {
                wv[i][j] = (random.nextDouble() - 0.5) * 0.1;
            }
        }



        //12. の初期化
        int dHidden = dModel * 2; // 表現力を高めるため、中間層の次元を倍の32次元に拡張！

// 3つの重み行列を定義
        double[][] wGate = new double[dModel][dHidden]; // ゲート用
        double[][] wUp   = new double[dModel][dHidden]; // アップ（バリュー）用
        double[][] wDown = new double[dHidden][dModel]; // ダウン用

// 3つの重み行列をすべてランダム初期化（対称性破壊）
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < dHidden; j++) {
                wGate[i][j] = (random.nextDouble() - 0.5) * 0.1;
                wUp[i][j]   = (random.nextDouble() - 0.5) * 0.1;
            }
        }
        for (int i = 0; i < dHidden; i++) {
            for (int j = 0; j < dModel; j++) {
                wDown[i][j] = (random.nextDouble() - 0.5) * 0.1;
            }
        }


        //13. の初期化
        int vocabSizeLocal = charToId.size();
        double[][] wOut = new double[dModel][vocabSizeLocal];
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < vocabSizeLocal; j++) {
                wOut[i][j] = (random.nextDouble() - 0.5) * 0.1;
            }
        }




        System.out.println("=== 学習開始 ===");

        for (int epoch = 0; epoch < epochs; epoch++) {

            // ★① エポックごとに、データセットから学習するセリフを1つ選ぶ（順番にローテーション）
            String currentText = dataset[epoch % dataset.length];

            int[] encoded = new int[currentText.length()];
            for (int i = 0; i < currentText.length(); i++) {
                encoded[i] = charToId.get(currentText.charAt(i));
            }

            // ★② このエポックでの文字数（シーケンス長）を定義する
            int seqLen = encoded.length;


            // 毎ターンの頭で勾配をゼロクリア
            for (double[] row : dWOut)   Arrays.fill(row, 0.0);
            for (double[] row : dWGate)  Arrays.fill(row, 0.0);
            for (double[] row : dWUp)    Arrays.fill(row, 0.0);
            for (double[] row : dWDown)  Arrays.fill(row, 0.0);
            for (double[] row : dWq)     Arrays.fill(row, 0.0);
            for (double[] row : dWk)     Arrays.fill(row, 0.0);
            for (double[] row : dWv)     Arrays.fill(row, 0.0);
            for (double[] row : dEmbeddingTable) Arrays.fill(row, 0.0);


            // 4. IDの配列から、16次元の座標を引っ張り出す（Embedding Lookup）
            double[][] inputEmbeddings = new double[encoded.length][vectorSize];


        for (int i = 0; i < encoded.length; i++) {
            int id = encoded[i]; // 文字のID
            // ごっそりコピー。
            System.arraycopy(embeddingTable[id], 0, inputEmbeddings[i], 0, vectorSize);
        }


        // 5. 位置エンコーディング（Positional Encoding）を足し合わせる


            double[] invFreq = new double[vectorSize];
            for (int j = 0; j < vectorSize; j++) {
                // 2 * (j / 2) は、0, 0, 2, 2, 4, 4... というように偶数ペアで同じ値になるおまじない
                double exponent = (double) (2 * (j / 2)) / vectorSize;
                invFreq[j] = 1.0 / Math.pow(10000.0, exponent);
            }

// ② メインのループ（Math.powが消えて、掛け算だけに進化！）
            for (int i = 0; i < encoded.length; i++) {
                for (int j = 0; j < vectorSize; j++) {
                    // 重い Math.pow の代わりに、あらかじめ用意した配列の値を「かける」だけにする！
                    double angle = i * invFreq[j];

                    if (j % 2 == 0) {
                        inputEmbeddings[i][j] += Math.sin(angle);
                    } else {
                        inputEmbeddings[i][j] += Math.cos(angle);
                    }
                }
            }



        // 6. Query (Q) の作成に特化した処理

        // 入力データ × Wq = Query (Q)
            double[][] query = new double[encoded.length][dModel];

            java.util.stream.IntStream.range(0, encoded.length).parallel().forEach(i -> {
                for (int j = 0; j < dModel; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < dModel; k++) {
                        sum += inputEmbeddings[i][k] * wq[k][j];
                    }
                    query[i][j] = sum;
                }
            });



        // 7. Key (K) を作るための重み行列（16次元 × 16次元）

        // 入力データ（inputEmbeddings） × Wk = Key (K)
            double[][] key = new double[encoded.length][dModel];

            java.util.stream.IntStream.range(0, encoded.length).parallel().forEach(i -> {
                for (int j = 0; j < dModel; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < dModel; k++) {
                        sum += inputEmbeddings[i][k] * wk[k][j];
                    }
                    key[i][j] = sum;
                }
            });

        // 8. Value (V) を作るための重み行列（16次元 × 16次元）

            // 入力データ（inputEmbeddings） × Wv = Value (V)
            double[][] value = new double[encoded.length][dModel];

            java.util.stream.IntStream.range(0, encoded.length).parallel().forEach(i -> {
                for (int j = 0; j < dModel; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < dModel; k++) {
                        sum += inputEmbeddings[i][k] * wv[k][j];
                    }
                    value[i][j] = sum;
                }
            });


        // 9. Causal Mask 付き Self-Attention のスコア計算
            double[][] attentionScores = new double[seqLen][seqLen];
            double scale = Math.sqrt(dModel);

            for (int i = 0; i < seqLen; i++) {
                // ① まず、未来の分（j > i）は全部 -1e9 で埋めておく
                java.util.Arrays.fill(attentionScores[i], -1e9);

                // ② 過去〜現在の自分自身（j <= i）の分だけ、必要なところだけ真面目に計算する
                for (int j = 0; j <= i; j++) {
                    double dotProduct = 0.0;
                    for (int k = 0; k < dModel; k++) {
                        dotProduct += query[i][k] * key[j][k];
                    }
                    attentionScores[i][j] = dotProduct / scale;
                }
            }

        // 10. Softmax関数でスコアを「注目度の割合（確率）」に変換する
        double[][] attentionWeights = new double[seqLen][seqLen];
            java.util.stream.IntStream.range(0, seqLen).parallel().forEach(i -> {
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



        // 11. Value (V) の情報を混ぜ合わせる（重み付き和の計算）
            double[][] attentionOutput = new double[seqLen][dModel];

            java.util.stream.IntStream.range(0, seqLen).parallel().forEach(i -> {
                for (int j = 0; j < dModel; j++) {
                    double weightedSum = 0.0;
                    for (int k = 0; k < seqLen; k++) {
                        weightedSum += attentionWeights[i][k] * value[k][j];
                    }
                    attentionOutput[i][j] = weightedSum;
                }
            });

        // ==========================================
// 12. 最先端LLM仕様：SwiGLU（Feed-Forward Network）の完全実装
// ==========================================

            double[][] ffnOutput = new double[seqLen][dModel];

// 外側のループを並列ストリームに置き換える
            IntStream.range(0, seqLen).parallel().forEach(i -> {
                // 各スレッドごとに独立した配列が作られるので安全！
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

                // ★ ここを dHidden から dModel に修正！
                for (int j = 0; j < dModel; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < dHidden; k++) {
                        sum += gatedValue[k] * wDown[k][j];
                    }
                    ffnOutput[i][j] = sum;
                }
            });

// 13. 最終出力層 ＆ 全位置（Causal LM）の損失計算
            int numPredictions = seqLen - 1;

// mapToDouble と sum() を使うことで、totalLoss への足し算を安全に並列処理できます！
            double totalLoss = IntStream.range(0, numPredictions).parallel().mapToDouble(i -> {
                int targetId = encoded[i + 1]; // 位置 i の次に来るべき文字のID

                // ロジット計算
                double[] logits = new double[vocabSizeLocal];
                for (int j = 0; j < vocabSizeLocal; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < dModel; k++) {
                        sum += ffnOutput[i][k] * wOut[k][j];
                    }
                    logits[j] = sum;
                }

                // オーバーフロー防止
                double maxLogit = logits[0];
                for (int j = 1; j < vocabSizeLocal; j++) {
                    if (logits[j] > maxLogit) maxLogit = logits[j];
                }

                // Log-Sum-Expトリックでクロスエントロピー損失を計算
                double targetLogit = logits[targetId];
                double logSumExp = 0.0;
                for (int j = 0; j < vocabSizeLocal; j++) {
                    logSumExp += Math.exp(logits[j] - maxLogit);
                }
                double logNormalizer = maxLogit + Math.log(logSumExp);

                double positionLoss = -(targetLogit - logNormalizer);

                // 各位置の損失を返す（自動で最後に全部足し合わされます）
                return positionLoss;
            }).sum();

            double finalLoss = totalLoss / numPredictions; // 平均損失
// ==========================================
// 14. 逆伝播（Backpropagation）の完全実装
// ==========================================

// 各文字位置ごとの隠れ状態（ffnOutput）に対する勾配を格納する配列
        double[][] dFfnOutput = new double[seqLen][dModel];

        numPredictions = seqLen - 1; // 4箇所

// --- [A] 出力層（wOut）の逆伝播 ---
        for (int i = 0; i < numPredictions; i++) {
            int targetId = encoded[i + 1];

            // この位置のロジットを再計算
            double[] logits = new double[vocabSizeLocal];
            for (int j = 0; j < vocabSizeLocal; j++) {
                double sum = 0.0;
                for (int k = 0; k < dModel; k++) {
                    sum += ffnOutput[i][k] * wOut[k][j];
                }
                logits[j] = sum;
            }

            // Softmax確率の再計算
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

            // Softmax + Cross-Entropy の勾配 (dLogits = probabilities - one_hot)
            double[] dLogits = new double[vocabSizeLocal];
            if (vocabSizeLocal >= 0) System.arraycopy(probs, 0, dLogits, 0, vocabSizeLocal);
            dLogits[targetId] -= 1.0;

            // wOut の勾配蓄積 ＆ ffnOutput への勾配の逆伝播
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

// --- [B] SwiGLU FFN層の逆伝播 ---
        for (int i = 0; i < seqLen; i++) {
            // 順伝播の中間状態（gate, up, gatedValue, SiLU用sigmoid）を正確に復元
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
                gate[j] = sum * sigmoid; // SiLU
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

            // dFfnOutput から wDown の勾配、および gatedValue への勾配を計算
            double[] dGatedValue = new double[dHidden];
            for (int k = 0; k < dHidden; k++) {
                double sum = 0.0;
                for (int j = 0; j < dModel; j++) {
                    sum += dFfnOutput[i][j] * wDown[k][j];
                    dWDown[k][j] += gatedValue[k] * dFfnOutput[i][j]; // wDown の勾配
                }
                dGatedValue[k] = sum;
            }

            // Hadamard積（gate * up）の逆伝播
            double[] dGate = new double[dHidden];
            double[] dUp = new double[dHidden];
            for (int k = 0; k < dHidden; k++) {
                dGate[k] = dGatedValue[k] * up[k];
                dUp[k] = dGatedValue[k] * gate[k];
            }

            // SiLU関数の微分を適用して wGate と wUp の勾配を計算
            for (int j = 0; j < dHidden; j++) {
                double sum = sumGateArr[j];
                double s = sigmoidArr[j];
                // SiLUの微分: d(silu(x))/dx = s + sum * s * (1 - s)
                double dSilu = s + (sum * s) * (1.0 - s);
                double dSumGate = dGate[j] * dSilu;

                for (int k = 0; k < dModel; k++) {
                    dWGate[k][j] += attentionOutput[i][k] * dSumGate;
                    dWUp[k][j]   += attentionOutput[i][k] * dUp[j];
                }
            }
        }

// --- [C] すべての重みパラメータの更新（勾配降下法） ---
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

            // ==========================================
            // 15. Self-Attention層の逆伝播（Backpropagation）の完全実装
            // ==========================================

            // FFN（SwiGLU）の中身を逆算して、Attentionの出力への正しい勾配を計算する
            double[][] dAttentionOutput = new double[seqLen][dModel];
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

            // ① Value (V) および Attention Weights（アテンション重み）の勾配を計算
            // attentionOutput[i][j] = sum_k (attentionWeights[i][k] * value[k][j])
            double[][] dAttentionWeights = new double[seqLen][seqLen];
            double[][] dValue = new double[seqLen][dModel];

            for (int i = 0; i < seqLen; i++) {
                for (int j = 0; j < dModel; j++) {
                    double gradOut = dAttentionOutput[i][j];
                    for (int k = 0; k < seqLen; k++) {
                        dAttentionWeights[i][k] += gradOut * value[k][j]; // 重みの勾配
                        dValue[k][j] += gradOut * attentionWeights[i][k]; // Valueの勾配
                    }
                }
            }

            // ② Softmax の逆伝播を通じて Attention Scores（スコア）の勾配を計算
            // （正しい数学的導出に基づく、シンプルで正確なSoftmaxの逆伝播です）
            double[][] dAttentionScores = new double[seqLen][seqLen];

            for (int i = 0; i < seqLen; i++) {
                // まず、行ごとに (dAttentionWeights * attentionWeights) の総和を計算しておく
                double dotSum = 0.0;
                for (int k = 0; k < seqLen; k++) {
                    dotSum += dAttentionWeights[i][k] * attentionWeights[i][k];
                }

                // 正しいSoftmaxの微分公式: dScore_j = y_j * (dWeight_j - dotSum)
                for (int j = 0; j < seqLen; j++) {
                    double y_j = attentionWeights[i][j];
                    dAttentionScores[i][j] = y_j * (dAttentionWeights[i][j] - dotSum);
                }
            }

            // ③ Causal Mask がかかっている部分のスコア勾配を完全に遮断（未来へのカンニング防止）
            for (int i = 0; i < seqLen; i++) {
                for (int j = 0; j < seqLen; j++) {
                    if (j > i) {
                        dAttentionScores[i][j] = 0.0;
                    }
                }
            }

            // ④ Query (Q) および Key (K) の勾配を計算
            // attentionScores[i][j] = (query[i] * key[j]) / scale
            double[][] dQuery = new double[seqLen][dModel];
            double[][] dKey = new double[seqLen][dModel];
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

        // ==========================================
// 16. W_q, W_k, W_v の逆伝播 ＆ Embedding層への接続
// ==========================================


// inputEmbeddings に対する勾配を格納する配列 (seqLen × dModel)
        double[][] dInputEmbeddings = new double[seqLen][dModel];

// --- [A] wq, wk, wv の勾配計算 ＆ inputEmbeddings への逆伝播 ---
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < dModel; j++) {
                double gQ = dQuery[i][j];
                double gK = dKey[i][j];
                double gV = dValue[i][j];

                for (int k = 0; k < dModel; k++) {
                    // 重み行列の勾配蓄積 (dW = input^T * dOutput)
                    dWq[k][j] += inputEmbeddings[i][k] * gQ;
                    dWk[k][j] += inputEmbeddings[i][k] * gK;
                    dWv[k][j] += inputEmbeddings[i][k] * gV;

                    // 入力ベクトル (inputEmbeddings) への勾配の蓄積 (dInput = dOutput * W^T)
                    dInputEmbeddings[i][k] += gQ * wq[k][j] + gK * wk[k][j] + gV * wv[k][j];
                }
            }
        }

// --- [B] wq, wk, wv の重みパラメータの更新（勾配降下法） ---
// ※ learningRate はすでに宣言されているため、型名（double等）はつけずに値を再利用します
        for (int i = 0; i < dModel; i++) {
            for (int j = 0; j < dModel; j++) {
                wq[i][j] -= learningRate * dWq[i][j];
                wk[i][j] -= learningRate * dWk[i][j];
                wv[i][j] -= learningRate * dWv[i][j];
            }
        }

// --- [C] embeddingTable（一番最初の辞書埋め込み）の勾配集約と更新 ---
// ※ 位置エンコーディングは定数（微分対象外）のため、伝わった勾配がそのまま embeddingTable へ反映されます

        for (int i = 0; i < seqLen; i++) {
            int id = encoded[i]; // その位置にある文字のID
            for (int j = 0; j < vectorSize; j++) {
                dEmbeddingTable[id][j] += dInputEmbeddings[i][j];
            }
        }

// embeddingTable のパラメータ更新
            for (int i = 0; i < vocabSize; i++) {
                for (int j = 0; j < vectorSize; j++) {
                    embeddingTable[i][j] -= learningRate * dEmbeddingTable[i][j];
                }
            }

            // 100エポックごとに進捗を表示
            if (epoch == 0 || (epoch + 1) % 500 == 0 || epoch == epochs - 1) {
                System.out.println("Epoch [" + (epoch + 1) + "/" + epochs + "] - Loss: " + finalLoss);
            }

        }





        System.out.println("\n=== 対話・文字生成テスト ===");


        Scanner scanner = new Scanner(System.in);

        while (true) {

// Scannerの準備
            System.out.print("AIへの入力文字をどうぞ: ");
            String userInput = scanner.nextLine(); // キーボードから入力を受け取る

// 入力された文字列を、1文字ずつIDの配列に変換する
            int[] genEncoded = new int[userInput.length()];
            for (int i = 0; i < userInput.length(); i++) {
                char c = userInput.charAt(i);
                // もし学習していない文字が含まれている場合の対策として getOrDefault を使うと安全です
                genEncoded[i] = charToId.getOrDefault(c, 0);
            }

            System.out.print("入力: " + userInput + "  生成結果: " + userInput);

            for (int step = 0; step < 30; step++) { // 残りの4文字分ループする
                int seqLen = genEncoded.length;

// 4. IDの配列から、16次元の座標を引っ張り出す（Embedding Lookup）
                double[][] inputEmbeddings = new double[genEncoded.length][vectorSize];


                for (int i = 0; i < genEncoded.length; i++) {
                    int id = genEncoded[i]; // 文字のID
                    // ごっそりコピー。
                    System.arraycopy(embeddingTable[id], 0, inputEmbeddings[i], 0, vectorSize);
                }


                // 5. 位置エンコーディング（Positional Encoding）を足し合わせる
                for (int i = 0; i < genEncoded.length; i++) { // 文字の順番（何番目か）
                    for (int j = 0; j < vectorSize; j++) { // 16次元のそれぞれの場所
                        // 位置によって変わる特殊な数値を計算する
                        double angle = i / Math.pow(10000.0, (double) (2 * (j / 2)) / vectorSize);

                        // 偶数番目の次元なら sin、奇数番目なら cos を足す
                        if (j % 2 == 0) {
                            inputEmbeddings[i][j] += Math.sin(angle);
                        } else {
                            inputEmbeddings[i][j] += Math.cos(angle);
                        }
                    }
                }


                // 6. Query (Q) の作成に特化した処理

                // 入力データ × Wq = Query (Q)
                double[][] query = new double[genEncoded.length][dModel];

                for (int i = 0; i < genEncoded.length; i++) {       // 各文字について
                    for (int j = 0; j < dModel; j++) {          // 16次元のそれぞれの場所
                        double sum = 0.0;
                        for (int k = 0; k < dModel; k++) {      // 行列の掛け算
                            sum += inputEmbeddings[i][k] * wq[k][j];
                        }
                        query[i][j] = sum; // これが文字ごとの「検索キーワード(Q)」！
                    }
                }


                // 7. Key (K) を作るための重み行列（16次元 × 16次元）

                // 入力データ（inputEmbeddings） × Wk = Key (K)
                double[][] key = new double[genEncoded.length][dModel];

                for (int i = 0; i < genEncoded.length; i++) {       // 各文字について
                    for (int j = 0; j < dModel; j++) {          // 16次元のそれぞれの場所
                        double sum = 0.0;
                        for (int k = 0; k < dModel; k++) {      // 行列の掛け算
                            sum += inputEmbeddings[i][k] * wk[k][j];
                        }
                        key[i][j] = sum; // これが文字ごとの「目印(K)」！
                    }
                }

                // 8. Value (V) を作るための重み行列（16次元 × 16次元）

                // 入力データ（inputEmbeddings） × Wv = Value (V)
                double[][] value = new double[genEncoded.length][dModel];

                for (int i = 0; i < genEncoded.length; i++) {       // 各文字について
                    for (int j = 0; j < dModel; j++) {          // 16次元のそれぞれの場所
                        double sum = 0.0;
                        for (int k = 0; k < dModel; k++) {      // 行列の掛け算
                            sum += inputEmbeddings[i][k] * wv[k][j];
                        }
                        value[i][j] = sum; // これが文字ごとの「中身(V)」！
                    }
                }


                // 9. Causal Mask 付き Self-Attention のスコア計算
                seqLen = genEncoded.length; // 文字数（5文字）
                double[][] attentionScores = new double[seqLen][seqLen];
                double scale = Math.sqrt(dModel);

                for (int i = 0; i < seqLen; i++) {
                    for (int j = 0; j < seqLen; j++) {
                        if (j > i) {
                            // 【Causal Mask】未来の文字へのカンニングを防止するため、極端に低い値で塞ぐ
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


                // 10. Softmax関数でスコアを「注目度の割合（確率）」に変換する
                double[][] attentionWeights = new double[seqLen][seqLen];

                for (int i = 0; i < seqLen; i++) {       // 各文字（行）ごとに計算

                    // ① オーバーフロー防止（数値が大きくなりすぎて壊れるのを防ぐおまじない）
                    // 行の中で一番大きい数字を見つける
                    double max = attentionScores[i][0];
                    for (int j = 1; j < seqLen; j++) {
                        if (attentionScores[i][j] > max) {
                            max = attentionScores[i][j];
                        }
                    }

                    // ② 指数関数（exp）を計算しつつ、行ごとの合計を求める
                    double sum = 0.0;
                    double[] expRow = new double[seqLen];
                    for (int j = 0; j < seqLen; j++) {
                        expRow[j] = Math.exp(attentionScores[i][j] - max); // maxを引くのが安全テクニック
                        sum += expRow[j];
                    }

                    // ③ 合計で割って、合計が「1.0（100%）」になるように正規化する
                    for (int j = 0; j < seqLen; j++) {
                        attentionWeights[i][j] = expRow[j] / sum;
                    }
                }


                // 11. Value (V) の情報を混ぜ合わせる（重み付き和の計算）
                double[][] attentionOutput = new double[seqLen][dModel];

                for (int i = 0; i < seqLen; i++) {       // 各文字（注目する側）について
                    for (int j = 0; j < dModel; j++) {   // 16次元のそれぞれの場所
                        double weightedSum = 0.0;
                        for (int k = 0; k < seqLen; k++) { // 他のすべての文字からのVをウェイト付きで足し合わせる
                            weightedSum += attentionWeights[i][k] * value[k][j];
                        }
                        attentionOutput[i][j] = weightedSum; // これが文脈を吸い込んだ新しいベクトル！
                    }
                }

                // ==========================================
// 12. 最先端LLM仕様：SwiGLU（Feed-Forward Network）の完全実装
// ==========================================


// FFNの出力格納用配列（5行 16列）
                double[][] ffnOutput = new double[seqLen][dModel];

// 各文字（シーケンスごと）に計算を実行
                for (int i = 0; i < seqLen; i++) {

                    // ステップ1: Gate Projection + Swish (SiLU) 関数の適用
                    double[] gate = new double[dHidden];
                    for (int j = 0; j < dHidden; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < dModel; k++) {
                            sum += attentionOutput[i][k] * wGate[k][j];
                        }
                        // Swish関数 (SiLU): x * sigmoid(x)
                        double sigmoid = 1.0 / (1.0 + Math.exp(-sum));
                        gate[j] = sum * sigmoid;
                    }

                    // ステップ2: Up Projection (特徴量の抽出)
                    double[] up = new double[dHidden];
                    for (int j = 0; j < dHidden; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < dModel; k++) {
                            sum += attentionOutput[i][k] * wUp[k][j];
                        }
                        up[j] = sum;
                    }

                    // ステップ3: Hadamard Product (ゲートとアップの要素ごとの掛け算)
                    double[] gatedValue = new double[dHidden];
                    for (int j = 0; j < dHidden; j++) {
                        gatedValue[j] = gate[j] * up[j];
                    }

                    // ステップ4: Down Projection (元の dModel 次元へ圧縮)
                    for (int j = 0; j < dModel; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < dHidden; k++) {
                            sum += gatedValue[k] * wDown[k][j];
                        }
                        ffnOutput[i][j] = sum; // これが本物のSwiGLUを通った強力なベクトル！
                    }


                }


                // 1. 最後の文字の位置（seqLen - 1）の予測ロジットを計算する
                int lastIdx = seqLen - 1;
                double[] logits = new double[vocabSizeLocal];
                for (int j = 0; j < vocabSizeLocal; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < dModel; k++) {
                        sum += ffnOutput[lastIdx][k] * wOut[k][j];
                    }
                    logits[j] = sum;
                }

                // 2. 一番確率が高い文字のIDを見つける（Argmax）
                int bestNextId = 0;
                double maxLogitVal = logits[0];
                for (int j = 1; j < vocabSizeLocal; j++) {
                    if (logits[j] > maxLogitVal) {
                        maxLogitVal = logits[j];
                        bestNextId = j;
                    }
                }

                // 3. 文字に変換して画面に表示する
                char predictedChar = idToChar.get(bestNextId);
                System.out.print(predictedChar);

                // 4. genEncoded の末尾に新しい文字を追加して、配列を1つ大きくする（次のループに繋げる）
                int[] nextGenEncoded = new int[seqLen + 1];
                System.arraycopy(genEncoded, 0, nextGenEncoded, 0, seqLen);
                nextGenEncoded[seqLen] = bestNextId;
                genEncoded = nextGenEncoded;


            }
            System.out.println();
        }



    }



}
