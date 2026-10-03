package com.mc1510ty.TekitouLLM;

import java.io.IOException;
import java.util.*;

public class Main {

    static void main() {

        long seed = 12345L;
        Random random = new Random(seed);

        // 1. 複数の会話パターン（データセット）を用意する

        String[] pretrainDataset = {
                "絶縁ゲートバイポーラトランジスタは半導体素子のひとつで、NPNPの4層からなりMOSゲートSCRまたはMOSゲートサイリスタと同じ構造でありながら、全動作領域でサイリスタ動作を完全に抑え込み、トランジスタ動作のみをさせるように設計した、MOSゲートで電流を制御するバイポーラトランジスタである。電力制御の用途で使用される。■",
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
                "デフォルトでは、TASC/ATO プラグインは手動の常用ブレーキで使用されるブレーキ指令を使用します。しかし実際の鉄道の TASC/ATO では、手動の常用ブレーキよりも細かくブレーキの強さを制御できるようになっていることがあります。車両性能とプラグインの設定を調節することで、そのような動作を再現することができます。■"
        };


        String[] chatDataset = {
                "U:こんにちは>A:こんにちは、良い天気ですね。■",
                "U:おはよう>A:おはようございます、今朝は何時に起きましたか？■",
                "U:こんばんは>A:こんばんは、月がきれいですね。■",
                "U:天気は？>A:私の学習データは常に更新されるわけではないので、リアルタイム性が重視される場合(今回の場合は今日の天気)、正しい情報を提供することができません。検索エンジンで'天気'などのワードで調べるのがはやいと思います。■",
                "U:今日の天気>A:学習データは最新ではないので、正しい情報を提供することができません。検索エンジンで'天気'などのワードで調べるのがはやいと思います。■",
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
                "U:BVE>A:'BVE'は、mackoy氏が主に個人で開発している、3DCGを用いたトレインシミュレーターのことです。最新バージョンは2020年9月23日にリリースされた、'BVE6'です。公式サイトのURLは、'https://bvets.net/'となっています。■",
                "U:LLM>A:LLMとは、簡単に言うと現代の会話できるAIのことです。■",
                "U:BVEとは>A:'BVE'とは、mackoy氏が個人で開発している、3DCG(3D描画)を用いたトレイン シミュレーターのことです。最新バージョンは2020/9/23にリリースされた、'BVE6'です。公式サイトのURLは、'https://bvets.net/'です。■",
        };

        boolean larned = false;
        ModelCheckpoint.LoadedModelData modeldata = null;

        try {
            modeldata = ModelCheckpoint.loadModel("a.omomi");
            System.out.println("モデルの変数の展開が完了しました、学習をスキップします");
            larned = true;
        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("重みの読み込みに失敗しました! 新規に学習を開始します。");
        }

        // 学習設定
        int numLayers = larned ? modeldata.numLayers : 6; // 層(レイヤー)の数
        int vectorSize = larned ? modeldata.vectorSize : 128; // 次元の数
        int maxSeqLen = 256; // 最大文字数
        int pretrainEpochs = 120; // 事前学習の回数
        int chatEpochs = 150;     // 指示ファインチューニングの回数
        int mixEpochs = 80;      // 最後に両方をごちゃ混ぜにするミックス学習

        double learningRate = 0.0005; // 学習率
        int batchSize = 12; // 並列数
        int targetVocabSize = 1200;

        // 設定や初期化のイメージ
        int num_heads = larned ? modeldata.num_heads : 8; // ヘッド数
        int head_size = vectorSize / num_heads;
        int epochs = pretrainEpochs + chatEpochs + mixEpochs;

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

        if (larned) {
            // ロード成功時は保存されていた辞書を復元
            tokenizer.tokenToId = modeldata.tokenToId;
            tokenizer.idToToken = modeldata.idToToken;
            tokenizer.merges = modeldata.merges;
            vocabSize = modeldata.vocabSize;
        } else {
            // ロード失敗時はこれまで通りデータセットを結合して訓練
            allDataset = new String[pretrainDataset.length + chatDataset.length];
            System.arraycopy(pretrainDataset, 0, allDataset, 0, pretrainDataset.length);
            System.arraycopy(chatDataset, 0, allDataset, pretrainDataset.length, chatDataset.length);

            tokenizer.train(allDataset, targetVocabSize, 2);

            System.out.println("\n作成されたサブワード:");
            tokenizer.tokenToId.entrySet().stream().forEach(entry -> {
                System.out.println("  [" + entry.getValue() + "] " + entry.getKey());
            });
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

        // --- グローバル勾配（全スレッドの結果を最終的に合算する場所） ---
        double[][] globalDWOut = new double[vectorSize][vocabSize];
        double[][][] globalDWGate = new double[numLayers][vectorSize][vectorSize * 2];
        double[][][] globalDWUp = new double[numLayers][vectorSize][vectorSize * 2];
        double[][][] globalDWDown = new double[numLayers][vectorSize * 2][vectorSize];
        double[][][][] globalDWq = new double[numLayers][num_heads][vectorSize][head_size];
        double[][][][] globalDWk = new double[numLayers][num_heads][vectorSize][head_size];
        double[][][][] globalDWv = new double[numLayers][num_heads][vectorSize][head_size];
        double[][] globalDEmbeddingTable = new double[vocabSize][vectorSize];

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

        double[][][] mDWGate = new double[numLayers][vectorSize][dHidden];
        double[][][] vDWGate = new double[numLayers][vectorSize][dHidden];
        double[][][] mDWUp = new double[numLayers][vectorSize][dHidden];
        double[][][] vDWUp = new double[numLayers][vectorSize][dHidden];
        double[][][] mDWDown = new double[numLayers][dHidden][vectorSize];
        double[][][] vDWDown = new double[numLayers][dHidden][vectorSize];

        double[][] globalDRmsWeightAttention = new double[numLayers][vectorSize];
        double[][] globalDRmsWeightFfn = new double[numLayers][vectorSize];

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

        double[][][][][] attentionScores = new double[batchSize][numLayers][num_heads][maxSeqLen][maxSeqLen];
        double[][][][][] attentionWeights = new double[batchSize][numLayers][num_heads][maxSeqLen][maxSeqLen];

        double[][][][] attentionOutput = new double[batchSize][numLayers][maxSeqLen][vectorSize];
        double[][][][] ffnOutput = new double[batchSize][numLayers][maxSeqLen][vectorSize];

        double[][][][] dFfnOutput = new double[batchSize][numLayers][maxSeqLen][vectorSize];
        double[][][][] dAttentionOutput = new double[batchSize][numLayers][maxSeqLen][vectorSize];
        double[][][][][] dAttentionWeights = new double[batchSize][numLayers][num_heads][maxSeqLen][maxSeqLen];
        double[][][][][] dValue = new double[batchSize][numLayers][maxSeqLen][num_heads][head_size];
        double[][][][][] dAttentionScores = new double[batchSize][numLayers][num_heads][maxSeqLen][maxSeqLen];
        double[][][][][] dQuery = new double[batchSize][numLayers][maxSeqLen][num_heads][head_size];
        double[][][][][] dKey = new double[batchSize][numLayers][maxSeqLen][num_heads][head_size];
        double[][][] dInputEmbeddings = new double[batchSize][maxSeqLen][vectorSize];

        double[][] mRmsWeightAttention = new double[numLayers][vectorSize];
        double[][] vRmsWeightAttention = new double[numLayers][vectorSize];
        double[][] mRmsWeightFfn = new double[numLayers][vectorSize];
        double[][] vRmsWeightFfn = new double[numLayers][vectorSize];

        if (!larned) {
            for (int i = 0; i < vocabSize; i++) {
                for (int j = 0; j < vectorSize; j++) {
                    embeddingTable[i][j] = (random.nextDouble() - 0.5);
                }
            }

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
                        double scale = Math.sqrt(head_size); // ★全体のdModelではなく、head_sizeの平方根にする

                        for (int h = 0; h < num_heads; h++) { // ヘッドのループを追加
                            for (int i = 0; i < seqLen; i++) {
                                // attentionScores もヘッドごとの次元を持つように形を変える必要があります
                                Arrays.fill(attentionScores[b][layer][h][i], 0, seqLen, -1e9);
                                for (int j = 0; j <= i; j++) {
                                    double dotProduct = 0.0;
                                    for (int k = 0; k < head_size; k++) { // 内積はヘッド内の次元（head_size）で計算
                                        dotProduct += query[b][layer][i][h][k] * key[b][layer][j][h][k];
                                    }
                                    attentionScores[b][layer][h][i][j] = dotProduct / scale;
                                }
                            }
                        }

// 10. Softmax（ヘッドごとに独立して計算）
                        for (int h = 0; h < num_heads; h++) { // ヘッドのループを追加
                            for (int i = 0; i < seqLen; i++) {
                                double max = attentionScores[b][layer][h][i][0];
                                for (int j = 1; j < seqLen; j++) {
                                    if (attentionScores[b][layer][h][i][j] > max) {
                                        max = attentionScores[b][layer][h][i][j];
                                    }
                                }
                                double sum = 0.0;
                                double[] expRow = new double[seqLen];
                                for (int j = 0; j < seqLen; j++) {
                                    expRow[j] = Math.exp(attentionScores[b][layer][h][i][j] - max);
                                    sum += expRow[j];
                                }
                                for (int j = 0; j < seqLen; j++) {
                                    attentionWeights[b][layer][h][i][j] = expRow[j] / sum;
                                }
                            }
                        }

// 11. Attention Output（各ヘッドの結果を Concatenate して元の次元に戻す ＋ 残差接続）
                        for (int i = 0; i < seqLen; i++) {
                            int outCol = 0; // 結合していくときのインデックス用
                            for (int h = 0; h < num_heads; h++) {
                                for (int j = 0; j < head_size; j++) {
                                    double weightedSum = 0.0;
                                    for (int k = 0; k < seqLen; k++) {
                                        weightedSum += attentionWeights[b][layer][h][i][k] * value[b][layer][k][h][j];
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
                                Arrays.fill(dAttentionWeights[b][l][h][i], 0.0);
                                Arrays.fill(dAttentionScores[b][l][h][i], 0.0);
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
                                for (int j = 0; j < head_size; j++) {
                                    int outCol = h * head_size + j;
                                    double gradOut = dAttentionOutput[b][l][i][outCol];
                                    for (int k = 0; k < seqLen; k++) {
                                        dAttentionWeights[b][l][h][i][k] += gradOut * value[b][l][k][h][j];
                                        dValue[b][l][k][h][j] += gradOut * attentionWeights[b][l][h][i][k];
                                    }
                                }
                            }
                        }

// 2. Softmax の逆伝播（ヘッドごとに計算）
                        for (int h = 0; h < num_heads; h++) {
                            for (int i = 0; i < seqLen; i++) {
                                double dotSumWeights = 0.0;
                                for (int k = 0; k < seqLen; k++) {
                                    dotSumWeights += dAttentionWeights[b][l][h][i][k] * attentionWeights[b][l][h][i][k];
                                }

                                for (int j = 0; j < seqLen; j++) {
                                    double y_j = attentionWeights[b][l][h][i][j];
                                    dAttentionScores[b][l][h][i][j] = y_j * (dAttentionWeights[b][l][h][i][j] - dotSumWeights);
                                }
                            }

                            // マスク処理（未来の情報を隠すための因果マスクの逆伝播対応）
                            for (int i = 0; i < seqLen; i++) {
                                for (int j = 0; j < seqLen; j++) {
                                    if (j > i) dAttentionScores[b][l][h][i][j] = 0.0;
                                }
                            }
                        }

// 3. Query と Key への勾配計算（ヘッドごと、scaleは head_size の平方根）
                        double scale = Math.sqrt(head_size);
                        for (int h = 0; h < num_heads; h++) {
                            for (int i = 0; i < seqLen; i++) {
                                for (int j = 0; j < seqLen; j++) {
                                    double dScore = dAttentionScores[b][l][h][i][j] / scale;
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

                // Lossの計算と表示
                if (epoch == 0 || (epoch + 1) % 1 == 0 || epoch == epochs - 1) {
                    double avgLoss = epochTotalLoss.sum() / batchSize;
                    System.out.println("Epoch [" + (epoch + 1) + "/" + epochs + "] - Loss: " + avgLoss);
                }
            }

            try {
                ModelCheckpoint.saveModel("a.omomi", numLayers, vectorSize, num_heads, dHidden, vocabSize, tokenizer.tokenToId, embeddingTable, wq, wk, wv, wGate, wUp, wDown, wOut, rmsWeightAttention, rmsWeightFfn,tokenizer.merges);
            } catch (IOException e) {
                e.printStackTrace();
                IO.println("重みの保存に失敗しました!");
            }

        }


        // --- 対話・文字生成テスト（バッチ対応済みの配列の 0 番目を使用） ---
        System.out.println("\n=== 対話・文字生成テスト ===");
        Scanner scanner = new Scanner(System.in);


        double[] logits = new double[vocabSizeLocal];
        double[] probs = new double[vocabSizeLocal];
        Integer[] vocabIndices = new Integer[vocabSizeLocal];
        for (int j = 0; j < vocabSizeLocal; j++) {
            vocabIndices[j] = j;
        }
        boolean[] keep = new boolean[vocabSizeLocal];


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

                    //RMSNorm
                    double[][] normedInput = new double[genSeqLen][dModel];
                    for (int i = 0; i < genSeqLen; i++) {
                        normedInput[i] = rmsNorm(currentLayerInput[i], rmsWeightAttention[l], 1e-5);
                    }

                    // 1. Q, K, V の計算（ヘッドごとに計算）
                    for (int h = 0; h < num_heads; h++) {
                        for (int i = 0; i < genSeqLen; i++) {
                            for (int j = 0; j < head_size; j++) {
                                double sumQ = 0.0, sumK = 0.0, sumV = 0.0;
                                for (int k = 0; k < dModel; k++) {
                                    sumQ += normedInput[i][k] * wq[l][h][k][j];
                                    sumK += normedInput[i][k] * wk[l][h][k][j];
                                    sumV += normedInput[i][k] * wv[l][h][k][j];
                                }
                                query[0][l][i][h][j] = sumQ;
                                key[0][l][i][h][j] = sumK;
                                value[0][l][i][h][j] = sumV;
                            }
                        }
                    }

                    // 2. アテンションスコアの計算（ヘッドごと、scaleは head_size の平方根）
                    double scale = Math.sqrt(head_size);
                    for (int h = 0; h < num_heads; h++) {
                        for (int i = 0; i < genSeqLen; i++) {
                            for (int j = 0; j < genSeqLen; j++) {
                                if (j > i) {
                                    attentionScores[0][l][h][i][j] = -1e9;
                                } else {
                                    double dotProduct = 0.0;
                                    for (int k = 0; k < head_size; k++) {
                                        dotProduct += query[0][l][i][h][k] * key[0][l][j][h][k];
                                    }
                                    attentionScores[0][l][h][i][j] = dotProduct / scale;
                                }
                            }
                        }
                    }

                    // 3. Softmax（ヘッドごとに計算）
                    for (int h = 0; h < num_heads; h++) {
                        for (int i = 0; i < genSeqLen; i++) {
                            double max = attentionScores[0][l][h][i][0];
                            for (int j = 1; j < genSeqLen; j++) {
                                if (attentionScores[0][l][h][i][j] > max) {
                                    max = attentionScores[0][l][h][i][j];
                                }
                            }

                            double sum = 0.0;
                            double[] expRow = new double[genSeqLen];
                            for (int j = 0; j < genSeqLen; j++) {
                                expRow[j] = Math.exp(attentionScores[0][l][h][i][j] - max);
                                sum += expRow[j];
                            }

                            for (int j = 0; j < genSeqLen; j++) {
                                attentionWeights[0][l][h][i][j] = expRow[j] / sum;
                            }
                        }
                    }

                    // 4. Attention Output（各ヘッドの結果を Concat して残差接続）
                    for (int i = 0; i < genSeqLen; i++) {
                        int outCol = 0;
                        for (int h = 0; h < num_heads; h++) {
                            for (int j = 0; j < head_size; j++) {
                                double weightedSum = 0.0;
                                for (int k = 0; k < genSeqLen; k++) {
                                    weightedSum += attentionWeights[0][l][h][i][k] * value[0][l][k][h][j];
                                }
                                attentionOutput[0][l][i][outCol++] = weightedSum;
                            }
                        }
                        // 残差接続（Residual Connection）
                        for (int j = 0; j < dModel; j++) {
                            attentionOutput[0][l][i][j] += currentLayerInput[i][j];
                        }
                    }


                    // ★【追加】FFNに入る前にRMSNormをかける
                    double[][] normedAttentionOutput = new double[genSeqLen][dModel];
                    for (int i = 0; i < genSeqLen; i++) {
                        normedAttentionOutput[i] = rmsNorm(attentionOutput[0][l][i], rmsWeightFfn[l], 1e-5);
                    }

                    //FFN
                    for (int i = 0; i < genSeqLen; i++) {
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

                int lastIdx = genSeqLen - 1;

                // 中身をリセットして使い回す
                Arrays.fill(logits, 0.0);
                for (int j = 0; j < vocabSizeLocal; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < dModel; k++) {
                        sum += ffnOutput[0][numLayers - 1][lastIdx][k] * wOut[k][j];
                    }
                    logits[j] = sum;
                }

                double temperature = 0.7;
                Arrays.fill(probs, 0.0); // 中身をリセット
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

                // ==========================================
                // Top-p（ニュークレアス・サンプリング）
                // ==========================================
                double topP = 0.9; // 確率の累計が90%に達するまでを候補にする

                // vocabIndices はあらかじめ作ってあるので、そのままソートに使う
                Arrays.sort(vocabIndices, (a, b) -> Double.compare(probs[b], probs[a]));

                double cumulativeProb = 0.0;
                Arrays.fill(keep, false); // 中身をリセット
                for (int j = 0; j < vocabSizeLocal; j++) {
                    int idx = vocabIndices[j];
                    cumulativeProb += probs[idx];
                    keep[idx] = true;
                    if (cumulativeProb >= topP) {
                        break; // 90%に達した時点で終了
                    }
                }

                // 残らなかったものの確率を0にして、残った分だけで確率を再計算
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
}