package com.mc1510ty.TekitouLLM;

import java.util.*;

public class Main {

    static void main() {

        long seed = 12345L;
        Random random = new Random(seed);

        // 1. 複数の会話パターン（データセット）を用意する

        String[] pretrainDataset = {
                "絶縁ゲートバイポーラトランジスタは半導体素子のひとつで、NPNPの4層からなりMOSゲートSCRまたはMOSゲートサイリスタ（英語版）と同じ構造でありながら、全動作領域でサイリスタ動作を完全に抑え込み、トランジスタ動作のみをさせるように設計した、MOSゲートで電流を制御するバイポーラトランジスタである。電力制御の用途で使用される。■",
                "この方針文書は、著作権の対象となっている著作物であって、一般公衆に開放されている屋外の場所、または一般公衆の見やすい屋外の場所に恒常的に設置された美術の著作物について、その著作権法上の扱いについて説明するとともに、当該著作物を被写体とする写真をウィキペディア日本語版において利用する際に守るべき事項を定めたものです。■",
                "被写体である美術著作物の題号と著作者名を画像ページに記載する。題号と著作者名が設置場所に表示されていない場合であっても、公表された文献に基づく調査を行い、それらが判明すれば記載する。一方、調査を行っても容易に判明しない場合は、記載する必要はない。また、著作者の意思により非公開としていると認められる場合には、記載してはならない。■",
                "画像は、被写体である美術著作物に密接に関連する事柄が記述されている1以上の記事（標準名前空間）で表示されなければならない。記事における画像表示は、画像のアップロード後すみやかに行い、将来、記事において画像を使用したいという漠然とした意思があるにすぎない状態では、画像のアップロードを避けること。■",
                "日米いずれの国の著作権法においても、著作物を被写体とする写真は、被写体である著作物の複製物または二次的著作物として扱われ、当該写真の利用に対しては、被写体である著作物の著作権の効力が及びます（日本国著作権法21条、28条、17 U.S.C. §102(a), §103）。したがって、日米両国の著作権法の下で、当該写真を被写体の著作物の著作権者の許諾を得ることなく適法に利用するには、両国の著作権法における著作権の制限規定に基づく必要があります。■",
                "東武10000系電車は、1983年（昭和58年）に登場した東武鉄道の通勤形電車。■",
                "ベースとなる10000型電車と、マイナーチェンジ車として1988年（昭和63年）に登場した10030型電車、VVVFインバータ制御試作車として1988年に登場した10080型電車にの3種に分類される。■",
                "Bve trainsim は、列車の運転をシミュレーションする Windows 用ゲームで、個人で制作しているフリーソフトです。■",
                "自分の好きな路線や車両を作成して運転することができます。ユーザーの皆さんが作成した多くの路線データや車両データがウェブ上に公開されています。データをダウンロードして組み込めば、運転できる路線や車両を増やすことができます。■",
                "Vulkan（ヴァルカン）は、クロノス・グループ（英: Khronos Group）が策定している、「PCやコンソールから、携帯電話や組込みプラットフォームに至るまで、あらゆるデバイスで使われる最先端のGPUに高効率かつクロスプラットフォーム型のアクセスを実現する、新世代のグラフィックス/コンピュートAPI」で、オープンスタンダード・ロイヤリティフリー・クロスプラットフォームとして提唱されている。Vulkan 1.0の仕様の公開日は2016年2月16日である。■",
                "グラフィックスハードウェア層に近いローレベル (low level) な制御を目的としており、これによりオーバーヘッドを低減し、ハードウェアの性能を限界まで引き出すことが可能となる。VulkanはAppleのMetalやマイクロソフトのDirect3D 12といった先発のローレベルAPIと競合するが、プラットフォーム独自の固有仕様ではなく、様々なデバイスやオペレーティングシステムをターゲットにできることが特徴である。■",
        };


        String[] chatDataset = {
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
                "U:OpenGLの代替は？>A:主に'Vulkan'が挙げられます。Windows専用であれば、'DirectX'なども選択肢としてあります。VulkanやDirectXについて質問がある場合は、なんでも言ってください！■",
                "U:こんにちは>A:こんにちは！今日は何をしますか？■",
                "U:こんばんは>A:こんばんは。今日は何がありましたか?■",
                "U:Windowsの最新バージョン>A:Microsoft Windowsの2026年10月2日時点での最新バージョンは、2021年10月5日にリリースされた'Windows 11'です。■",
                "U:BVE>A:'BVE'は、mackoy氏が主に個人で開発している、3DCGを用いたトレインシミュレーターのことです。最新バージョンは2020年9月23日にリリースされた、'BVE6'です。公式サイトのURLは、'https://bvets.net/'となっています。■"
        };


        // 学習設定
        int numLayers = 12; //層(レイヤー)の数
        int maxSeqLen = 256; //最大文字数(ふやしすぎるとメモリ爆発)
        int vectorSize = 96; //次元の数
        int pretrainEpochs = 80; // 事前学習の回数
        int chatEpochs = 150;     // 指示ファインチューニングの回数
        int mixEpochs = 100;      // 最後に両方をごちゃ混ぜにするミックス学習

        double learningRate = 0.0005; //学習率
        int batchSize = 16; //並列数(String[]の中身がこの数字以下だと並列化が少なくなる)




        int epochs = pretrainEpochs + chatEpochs + mixEpochs;


        // ★ 追加：AdamWのハイパーパラメータ
        double beta1 = 0.9;
        double beta2 = 0.999;
        double eps = 1e-8;
        double weightDecay = 0.01; // 重み減衰（過学習を抑える強さ）
        int adamStep = 0; // 更新ステップ数カウンター



        // 2. BPEトークナイザーの初期化と学習
        // 事前学習用と対話用の両方の言葉をトークナイザーに覚えさせるために結合します
        String[] allDataset = new String[pretrainDataset.length + chatDataset.length];
        System.arraycopy(pretrainDataset, 0, allDataset, 0, pretrainDataset.length);
        System.arraycopy(chatDataset, 0, allDataset, pretrainDataset.length, chatDataset.length);

        SimpleTokenizer tokenizer = new SimpleTokenizer();
        // 結合した allDataset を使ってトークナイザーを訓練
        tokenizer.train(allDataset, 1000, 2);

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

        double[][][] wq = new double[numLayers][dModel][dModel];
        double[][][] wk = new double[numLayers][dModel][dModel];
        double[][][] wv = new double[numLayers][dModel][dModel];
        double[][][] wGate = new double[numLayers][dModel][dHidden];
        double[][][] wUp = new double[numLayers][dModel][dHidden];
        double[][][] wDown = new double[numLayers][dHidden][dModel];
        double[][] wOut = new double[dModel][vocabSizeLocal];

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
                    wUp[l][i][j] = (random.nextDouble() - 0.5) * 0.1;
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

        // --- グローバル勾配（全スレッドの結果を最終的に合算する場所） ---
        double[][] globalDWOut = new double[vectorSize][vocabSize];
        double[][][] globalDWGate = new double[numLayers][vectorSize][vectorSize * 2];
        double[][][] globalDWUp = new double[numLayers][vectorSize][vectorSize * 2];
        double[][][] globalDWDown = new double[numLayers][vectorSize * 2][vectorSize];
        double[][][] globalDWq = new double[numLayers][vectorSize][vectorSize];
        double[][][] globalDWk = new double[numLayers][vectorSize][vectorSize];
        double[][][] globalDWv = new double[numLayers][vectorSize][vectorSize];
        double[][] globalDEmbeddingTable = new double[vocabSize][vectorSize];



        // --- AdamW用のモーメント配列（1次モーメント m, 2次モーメント v）の用意 ---
        double[][] mDWOut = new double[vectorSize][vocabSize];
        double[][] vDWOut = new double[vectorSize][vocabSize];

        double[][] mDEmbeddingTable = new double[vocabSize][vectorSize];
        double[][] vDEmbeddingTable = new double[vocabSize][vectorSize];

        double[][][] mDWq = new double[numLayers][vectorSize][vectorSize];
        double[][][] vDWq = new double[numLayers][vectorSize][vectorSize];
        double[][][] mDWk = new double[numLayers][vectorSize][vectorSize];
        double[][][] vDWk = new double[numLayers][vectorSize][vectorSize];
        double[][][] mDWv = new double[numLayers][vectorSize][vectorSize];
        double[][][] vDWv = new double[numLayers][vectorSize][vectorSize];

        double[][][] mDWGate = new double[numLayers][vectorSize][dHidden];
        double[][][] vDWGate = new double[numLayers][vectorSize][dHidden];
        double[][][] mDWUp = new double[numLayers][vectorSize][dHidden];
        double[][][] vDWUp = new double[numLayers][vectorSize][dHidden];

        double[][][] mDWDown = new double[numLayers][dHidden][vectorSize];
        double[][][] vDWDown = new double[numLayers][dHidden][vectorSize];




        // --- ローカル勾配（各スレッドが自分専用に使う作業机：一番左に batchSize が付く） ---
        double[][][] localDWOut = new double[batchSize][vectorSize][vocabSize];
        double[][][][] localDWGate = new double[batchSize][numLayers][vectorSize][vectorSize * 2];
        double[][][][] localDWUp = new double[batchSize][numLayers][vectorSize][vectorSize * 2];
        double[][][][] localDWDown = new double[batchSize][numLayers][vectorSize * 2][vectorSize];
        double[][][][] localDWq = new double[batchSize][numLayers][vectorSize][vectorSize];
        double[][][][] localDWk = new double[batchSize][numLayers][vectorSize][vectorSize];
        double[][][][] localDWv = new double[batchSize][numLayers][vectorSize][vectorSize];
        double[][][] localDEmbeddingTable = new double[batchSize][vocabSize][vectorSize];

        double[] invFreq = new double[vectorSize];
        for (int j = 0; j < vectorSize; j++) {
            double exponent = (double) (2 * (j / 2)) / vectorSize;
            invFreq[j] = 1.0 / Math.pow(10000.0, exponent);
        }

        double[][][] inputEmbeddings = new double[batchSize][maxSeqLen][vectorSize];

        // --- バッチ対応させた作業スペース ---
        double[][][][] query = new double[batchSize][numLayers][maxSeqLen][dModel];
        double[][][][] key = new double[batchSize][numLayers][maxSeqLen][dModel];
        double[][][][] value = new double[batchSize][numLayers][maxSeqLen][dModel];
        double[][][][] attentionScores = new double[batchSize][numLayers][maxSeqLen][maxSeqLen];
        double[][][][] attentionWeights = new double[batchSize][numLayers][maxSeqLen][maxSeqLen];
        double[][][][] attentionOutput = new double[batchSize][numLayers][maxSeqLen][dModel];
        double[][][][] ffnOutput = new double[batchSize][numLayers][maxSeqLen][dModel];

        double[][][][] dFfnOutput = new double[batchSize][numLayers][maxSeqLen][dModel];
        double[][][][] dAttentionOutput = new double[batchSize][numLayers][maxSeqLen][dModel];
        double[][][][] dAttentionWeights = new double[batchSize][numLayers][maxSeqLen][maxSeqLen];
        double[][][][] dValue = new double[batchSize][numLayers][maxSeqLen][dModel];
        double[][][][] dAttentionScores = new double[batchSize][numLayers][maxSeqLen][maxSeqLen];
        double[][][][] dQuery = new double[batchSize][numLayers][maxSeqLen][dModel];
        double[][][][] dKey = new double[batchSize][numLayers][maxSeqLen][dModel];
        double[][][] dInputEmbeddings = new double[batchSize][maxSeqLen][dModel];


        System.out.println("=== 学習開始 ===");

        int epochstatus = 0;

        for (int epoch = 0; epoch < epochs; epoch++) {


            String[] currentDataset;
            if (epoch < pretrainEpochs) {
                currentDataset = pretrainDataset;
                if (epochstatus == 0) {
                    IO.println("=====事前学習=====");
                    epochstatus = 1;
                }
            } else if (epoch < pretrainEpochs + chatEpochs) {
                currentDataset = chatDataset;
                if (epochstatus == 1) {
                    IO.println("=====対話学習=====");
                    epochstatus = 2;
                }
            } else {
                currentDataset = allDataset;
                if (epochstatus == 2) {
                    IO.println("=====混合学習=====");
                    epochstatus = 3;
                }
            }

            Integer[] indices = new Integer[currentDataset.length];
            for (int i = 0; i < indices.length; i++) {
                indices[i] = i;
            }
            List<Integer> indexList = Arrays.asList(indices);
            Collections.shuffle(indexList, random); // 最初に用意した random を使います
            Integer[] shuffledIndices = indexList.toArray(new Integer[0]);




            // --- 1. グローバル勾配のゼロクリア ---
            for (double[] row : globalDWOut) Arrays.fill(row, 0.0);
            for (double[] row : globalDEmbeddingTable) Arrays.fill(row, 0.0);
            for (double[][] matrix : globalDWGate) for (double[] row : matrix) Arrays.fill(row, 0.0);
            for (double[][] matrix : globalDWUp) for (double[] row : matrix) Arrays.fill(row, 0.0);
            for (double[][] matrix : globalDWDown) for (double[] row : matrix) Arrays.fill(row, 0.0);
            for (double[][] matrix : globalDWq) for (double[] row : matrix) Arrays.fill(row, 0.0);
            for (double[][] matrix : globalDWk) for (double[] row : matrix) Arrays.fill(row, 0.0);
            for (double[][] matrix : globalDWv) for (double[] row : matrix) Arrays.fill(row, 0.0);

            // --- 2. 各スレッド（ローカル）の勾配もゼロクリア ---
            for (int b = 0; b < batchSize; b++) {
                for (double[] row : localDWOut[b]) Arrays.fill(row, 0.0);
                for (double[] row : localDEmbeddingTable[b]) Arrays.fill(row, 0.0);
                for (double[][] matrix : localDWGate[b]) for (double[] row : matrix) Arrays.fill(row, 0.0);
                for (double[][] matrix : localDWUp[b]) for (double[] row : matrix) Arrays.fill(row, 0.0);
                for (double[][] matrix : localDWDown[b]) for (double[] row : matrix) Arrays.fill(row, 0.0);
                for (double[][] matrix : localDWq[b]) for (double[] row : matrix) Arrays.fill(row, 0.0);
                for (double[][] matrix : localDWk[b]) for (double[] row : matrix) Arrays.fill(row, 0.0);
                for (double[][] matrix : localDWv[b]) for (double[] row : matrix) Arrays.fill(row, 0.0);
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

                    // 6. Query (Q)
                    for (int i = 0; i < seqLen; i++) {
                        for (int j = 0; j < dModel; j++) {
                            double sum = 0.0;
                            for (int k = 0; k < dModel; k++) {
                                sum += currentLayerInput[i][k] * wq[layer][k][j];
                            }
                            query[b][layer][i][j] = sum;
                        }
                    }

                    // 7. Key (K)
                    for (int i = 0; i < seqLen; i++) {
                        for (int j = 0; j < dModel; j++) {
                            double sum = 0.0;
                            for (int k = 0; k < dModel; k++) {
                                sum += currentLayerInput[i][k] * wk[layer][k][j];
                            }
                            key[b][layer][i][j] = sum;
                        }
                    }

                    // 8. Value (V)
                    for (int i = 0; i < seqLen; i++) {
                        for (int j = 0; j < dModel; j++) {
                            double sum = 0.0;
                            for (int k = 0; k < dModel; k++) {
                                sum += currentLayerInput[i][k] * wv[layer][k][j];
                            }
                            value[b][layer][i][j] = sum;
                        }
                    }

                    // 9. Attention スコア
                    double scale = Math.sqrt(dModel);
                    for (int i = 0; i < seqLen; i++) {
                        Arrays.fill(attentionScores[b][layer][i], 0, seqLen, -1e9);
                        for (int j = 0; j <= i; j++) {
                            double dotProduct = 0.0;
                            for (int k = 0; k < dModel; k++) {
                                dotProduct += query[b][layer][i][k] * key[b][layer][j][k];
                            }
                            attentionScores[b][layer][i][j] = dotProduct / scale;
                        }
                    }

                    // 10. Softmax
                    for (int i = 0; i < seqLen; i++) {
                        double max = attentionScores[b][layer][i][0];
                        for (int j = 1; j < seqLen; j++) {
                            if (attentionScores[b][layer][i][j] > max) {
                                max = attentionScores[b][layer][i][j];
                            }
                        }
                        double sum = 0.0;
                        double[] expRow = new double[seqLen];
                        for (int j = 0; j < seqLen; j++) {
                            expRow[j] = Math.exp(attentionScores[b][layer][i][j] - max);
                            sum += expRow[j];
                        }
                        for (int j = 0; j < seqLen; j++) {
                            attentionWeights[b][layer][i][j] = expRow[j] / sum;
                        }
                    }

                    // 11. Attention Output
                    for (int i = 0; i < seqLen; i++) {
                        for (int j = 0; j < dModel; j++) {
                            double weightedSum = 0.0;
                            for (int k = 0; k < seqLen; k++) {
                                weightedSum += attentionWeights[b][layer][i][k] * value[b][layer][k][j];
                            }
                            attentionOutput[b][layer][i][j] = weightedSum + currentLayerInput[i][j];
                        }
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
                                double val = attentionOutput[b][layer][i][k];
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
                            ffnOutput[b][layer][i][j] = sum + attentionOutput[b][layer][i][j];
                        }
                    }
                }

                // 13. 逆伝播ワークスペース初期化
                for (int l = 0; l < numLayers; l++) {
                    for (int i = 0; i < seqLen; i++) {
                        Arrays.fill(dFfnOutput[b][l][i], 0.0);
                        Arrays.fill(dAttentionOutput[b][l][i], 0.0);
                        Arrays.fill(dQuery[b][l][i], 0.0);
                        Arrays.fill(dKey[b][l][i], 0.0);
                        Arrays.fill(dValue[b][l][i], 0.0);
                    }
                    for (int i = 0; i < seqLen; i++) {
                        Arrays.fill(dAttentionWeights[b][l][i], 0.0);
                        Arrays.fill(dAttentionScores[b][l][i], 0.0);
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

                    for (int i = 0; i < seqLen; i++) {
                        for (int j = 0; j < dModel; j++) {
                            double gradOut = dAttentionOutput[b][l][i][j];
                            for (int k = 0; k < seqLen; k++) {
                                dAttentionWeights[b][l][i][k] += gradOut * value[b][l][k][j];
                                dValue[b][l][k][j] += gradOut * attentionWeights[b][l][i][k];
                            }
                        }
                    }

                    for (int i = 0; i < seqLen; i++) {
                        double dotSumWeights = 0.0;
                        for (int k = 0; k < seqLen; k++) {
                            dotSumWeights += dAttentionWeights[b][l][i][k] * attentionWeights[b][l][i][k];
                        }

                        for (int j = 0; j < seqLen; j++) {
                            double y_j = attentionWeights[b][l][i][j];
                            dAttentionScores[b][l][i][j] = y_j * (dAttentionWeights[b][l][i][j] - dotSumWeights);
                        }
                    }

                    for (int i = 0; i < seqLen; i++) {
                        for (int j = 0; j < seqLen; j++) {
                            if (j > i) dAttentionScores[b][l][i][j] = 0.0;
                        }
                    }

                    double scale = Math.sqrt(dModel);
                    for (int i = 0; i < seqLen; i++) {
                        for (int j = 0; j < seqLen; j++) {
                            double dScore = dAttentionScores[b][l][i][j] / scale;
                            for (int k = 0; k < dModel; k++) {
                                dQuery[b][l][i][k] += dScore * key[b][l][j][k];
                                dKey[b][l][j][k] += dScore * query[b][l][i][k];
                            }
                        }
                    }

                    for (int i = 0; i < seqLen; i++) {
                        for (int k = 0; k < dModel; k++) {
                            double gradOut = dAttentionOutput[b][l][i][k];
                            if (l > 0) {
                                dFfnOutput[b][l - 1][i][k] += gradOut;
                            } else {
                                dInputEmbeddings[b][i][k] += gradOut;
                            }
                        }
                    }

                    for (int i = 0; i < seqLen; i++) {
                        for (int j = 0; j < dModel; j++) {
                            double gQ = dQuery[b][l][i][j];
                            double gV = dValue[b][l][i][j];

                            for (int k = 0; k < dModel; k++) {
                                localDWq[b][l][k][j] += currentLayerInput[i][k] * gQ;
                                localDWv[b][l][k][j] += currentLayerInput[i][k] * gV;

                                double gradInputVal = gQ * wq[l][k][j] + gV * wv[l][k][j];
                                if (l > 0) {
                                    dFfnOutput[b][l - 1][i][k] += gradInputVal;
                                } else {
                                    dInputEmbeddings[b][i][k] += gradInputVal;
                                }
                            }
                        }
                    }

                    for (int j = 0; j < seqLen; j++) {
                        for (int k = 0; k < dModel; k++) {
                            double gK = dKey[b][l][j][k];
                            for (int m = 0; m < dModel; m++) {
                                localDWk[b][l][m][k] += currentLayerInput[j][m] * gK;
                                double gradInputValK = gK * wk[l][m][k];
                                if (l > 0) {
                                    dFfnOutput[b][l - 1][j][m] += gradInputValK;
                                } else {
                                    dInputEmbeddings[b][j][m] += gradInputValK;
                                }
                            }
                        }
                    }
                }

                // 埋め込み層への勾配蓄積
                for (int i = 0; i < seqLen; i++) {
                    int id = encoded[i];
                    for (int j = 0; j < vectorSize; j++) {
                        localDEmbeddingTable[b][id][j] += dInputEmbeddings[b][i][j];
                    }
                }
            });

            // --- 4. 16個のローカル勾配をグローバル勾配に合算する ---
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
                    for (int i = 0; i < vectorSize; i++) {
                        for (int j = 0; j < vectorSize; j++) {
                            globalDWq[l][i][j] += localDWq[b][l][i][j];
                            globalDWk[l][i][j] += localDWk[b][l][i][j];
                            globalDWv[l][i][j] += localDWv[b][l][i][j];
                        }
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
                for (int i = 0; i < vectorSize; i++) {
                    for (int j = 0; j < vectorSize; j++) {
                        globalDWq[l][i][j] *= invBatchSize;
                        globalDWk[l][i][j] *= invBatchSize;
                        globalDWv[l][i][j] *= invBatchSize;
                    }
                }
                for (int i = 0; i < vectorSize; i++) {
                    for (int j = 0; j < dHidden; j++) {
                        globalDWGate[l][i][j] *= invBatchSize;
                        globalDWUp[l][i][j]   *= invBatchSize;
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
                // wq
                for (int i = 0; i < dModel; i++) {
                    for (int j = 0; j < dModel; j++) {
                        double g = globalDWq[l][i][j];
                        mDWq[l][i][j] = beta1 * mDWq[l][i][j] + (1.0 - beta1) * g;
                        vDWq[l][i][j] = beta2 * vDWq[l][i][j] + (1.0 - beta2) * (g * g);
                        double mHat = mDWq[l][i][j] / correction1;
                        double vHat = vDWq[l][i][j] / correction2;
                        wq[l][i][j] = wq[l][i][j] - learningRate * weightDecay * wq[l][i][j] - learningRate * mHat / (Math.sqrt(vHat) + eps);
                    }
                }
                // wk
                for (int i = 0; i < dModel; i++) {
                    for (int j = 0; j < dModel; j++) {
                        double g = globalDWk[l][i][j];
                        mDWk[l][i][j] = beta1 * mDWk[l][i][j] + (1.0 - beta1) * g;
                        vDWk[l][i][j] = beta2 * vDWk[l][i][j] + (1.0 - beta2) * (g * g);
                        double mHat = mDWk[l][i][j] / correction1;
                        double vHat = vDWk[l][i][j] / correction2;
                        wk[l][i][j] = wk[l][i][j] - learningRate * weightDecay * wk[l][i][j] - learningRate * mHat / (Math.sqrt(vHat) + eps);
                    }
                }
                // wv
                for (int i = 0; i < dModel; i++) {
                    for (int j = 0; j < dModel; j++) {
                        double g = globalDWv[l][i][j];
                        mDWv[l][i][j] = beta1 * mDWv[l][i][j] + (1.0 - beta1) * g;
                        vDWv[l][i][j] = beta2 * vDWv[l][i][j] + (1.0 - beta2) * (g * g);
                        double mHat = mDWv[l][i][j] / correction1;
                        double vHat = vDWv[l][i][j] / correction2;
                        wv[l][i][j] = wv[l][i][j] - learningRate * weightDecay * wv[l][i][j] - learningRate * mHat / (Math.sqrt(vHat) + eps);
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
            }

            // Lossの計算と表示
            if (epoch == 0 || (epoch + 1) % 1 == 0 || epoch == epochs - 1) {
                double avgLoss = epochTotalLoss.sum() / batchSize;
                System.out.println("Epoch [" + (epoch + 1) + "/" + epochs + "] - Loss: " + avgLoss);
            }
        }


        // --- 対話・文字生成テスト（バッチ対応済みの配列の 0 番目を使用） ---
        System.out.println("\n=== 対話・文字生成テスト ===");
        Scanner scanner = new Scanner(System.in);

        while (true) {
            System.out.print("AIへの入力文字をどうぞ: ");
            String userInput = scanner.nextLine();

            String prompt = "U:" + userInput + ">A:";
            List<Integer> promptEncodedList = tokenizer.encode(prompt);
            int genSeqLen = promptEncodedList.size();
            int[] genEncoded = new int[genSeqLen];
            for (int i = 0; i < genSeqLen; i++) {
                genEncoded[i] = promptEncodedList.get(i);
            }

            System.out.print("入力: " + userInput + "  生成結果: " + prompt);

            for (int step = 0; step < 300; step++) {
                genSeqLen = genEncoded.length;

                // 推論時はバッチの [0] 番目の作業スペースを利用する
                for (int i = 0; i < genSeqLen; i++) {
                    int id = genEncoded[i];
                    System.arraycopy(embeddingTable[id], 0, inputEmbeddings[0][i], 0, vectorSize);
                }

                for (int i = 0; i < genSeqLen; i++) {
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

                    for (int i = 0; i < genSeqLen; i++) {
                        for (int j = 0; j < dModel; j++) {
                            double sumQ = 0.0, sumK = 0.0, sumV = 0.0;
                            for (int k = 0; k < dModel; k++) {
                                sumQ += currentLayerInput[i][k] * wq[l][k][j];
                                sumK += currentLayerInput[i][k] * wk[l][k][j];
                                sumV += currentLayerInput[i][k] * wv[l][k][j];
                            }
                            query[0][l][i][j] = sumQ;
                            key[0][l][i][j] = sumK;
                            value[0][l][i][j] = sumV;
                        }
                    }

                    double scale = Math.sqrt(dModel);
                    for (int i = 0; i < genSeqLen; i++) {
                        for (int j = 0; j < genSeqLen; j++) {
                            if (j > i) {
                                attentionScores[0][l][i][j] = -1e9;
                            } else {
                                double dotProduct = 0.0;
                                for (int k = 0; k < dModel; k++) {
                                    dotProduct += query[0][l][i][k] * key[0][l][j][k];
                                }
                                attentionScores[0][l][i][j] = dotProduct / scale;
                            }
                        }
                    }

                    for (int i = 0; i < genSeqLen; i++) {
                        double max = attentionScores[0][l][i][0];
                        for (int j = 1; j < genSeqLen; j++) {
                            if (attentionScores[0][l][i][j] > max) {
                                max = attentionScores[0][l][i][j];
                            }
                        }

                        double sum = 0.0;
                        double[] expRow = new double[genSeqLen];
                        for (int j = 0; j < genSeqLen; j++) {
                            expRow[j] = Math.exp(attentionScores[0][l][i][j] - max);
                            sum += expRow[j];
                        }

                        for (int j = 0; j < genSeqLen; j++) {
                            attentionWeights[0][l][i][j] = expRow[j] / sum;
                        }
                    }

                    for (int i = 0; i < genSeqLen; i++) {
                        for (int j = 0; j < dModel; j++) {
                            double weightedSum = 0.0;
                            for (int k = 0; k < genSeqLen; k++) {
                                weightedSum += attentionWeights[0][l][i][k] * value[0][l][k][j];
                            }
                            attentionOutput[0][l][i][j] = weightedSum + currentLayerInput[i][j];
                        }
                    }

                    for (int i = 0; i < genSeqLen; i++) {
                        double[] gate = new double[dHidden];
                        double[] up = new double[dHidden];
                        for (int j = 0; j < dHidden; j++) {
                            double sumGate = 0.0, sumUp = 0.0;
                            for (int k = 0; k < dModel; k++) {
                                double val = attentionOutput[0][l][i][k];
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

                int lastIdx = genSeqLen - 1;
                double[] logits = new double[vocabSizeLocal];
                for (int j = 0; j < vocabSizeLocal; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < dModel; k++) {
                        sum += ffnOutput[0][numLayers - 1][lastIdx][k] * wOut[k][j];
                    }
                    logits[j] = sum;
                }

                double temperature = 0.7;
                double[] probs = new double[vocabSizeLocal];
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

                double r = random.nextDouble();
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