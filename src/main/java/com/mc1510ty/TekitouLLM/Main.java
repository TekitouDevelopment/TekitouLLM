package com.mc1510ty.TekitouLLM;

import java.nio.charset.StandardCharsets;
import java.util.Random;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        int mozisuuseigenn = 9999;

// ユーザーの入力の例
        String[] userInputs = {
                "おはよう",
                "おは",
                "おはよ",
                "こんにちは",
                "え？",
                "眠い",
                "わかった",
                "OK",
                "こん",
                "。。。",
                "眠いよ",
                "おやすみ",
        };

// AIの回答の例
        String[] aiResponses = {
                "おはようございます！",
                "おはようございます！",
                "おはようございます！",
                "こんにちは！",
                "驚いたことなどがありましたか？",
                "寝るのがおすすめです。",
                "ご理解いただきありがとうございます！",
                "ご理解いただきありがとうございます！",
                "こんにちは～",
                "何かありましたか？",
                "寝るのがおすすめです",
                "おやすみなさい！",
        };

// 🌟 プログラム側で「入力 + $ + 回答 + $」の形に自動で合体させる！
        String[] texts = new String[userInputs.length];
        for (int i = 0; i < userInputs.length; i++) {
            texts[i] = userInputs[i] + "$" + aiResponses[i] + "$";
        }
        String[] processedTexts = texts;

        // ボキャブラリーの抽出
        String uniqueChars = "";
        for (String t : texts) {
            for (char c : t.toCharArray()) {
                if (uniqueChars.indexOf(c) == -1) {
                    uniqueChars += c;
                }
            }
        }
        char[] vocab = uniqueChars.toCharArray();
        int V = vocab.length;

        // ハイパーパラメータ
        int H = 64; // 隠れ層（記憶のバッグ）のサイズ
        double learningRate = 0.01;
        int epochs = 1000;

        // 重みとバイアスの初期化
        Random rand = new Random(42);

        double[][] Wxh = new double[H][V]; // 入力 -> 隠れ層
        double[][] Whh = new double[H][H]; // 隠れ層 -> 隠れ層（時間のループ！）
        double[][] Why = new double[V][H]; // 隠れ層 -> 出力
        double[] bh = new double[H];       // 隠れ層のバイアス
        double[] by = new double[V];       // 出力のバイアス

        for (int i = 0; i < H; i++) {
            for (int j = 0; j < V; j++) {
                Wxh[i][j] = (rand.nextDouble() - 0.5) * 0.1;
            }
            for (int j = 0; j < H; j++) {
                Whh[i][j] = (rand.nextDouble() - 0.5) * 0.1;
            }
            bh[i] = 0;
        }
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < H; j++) {
                Why[i][j] = (rand.nextDouble() - 0.5) * 0.1;
            }
            by[i] = 0;
        }

        // 学習ループ (BPTT: Backpropagation Through Time)
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (String t : texts) {
                int n = t.length();
                if (n < 2) continue;

                double[][] hs = new double[n][H];
                double[][] ps = new double[n][V];
                int[] inputs = new int[n];
                int[] targets = new int[n];

                for (int i = 0; i < n; i++) {
                    char c = t.charAt(i);
                    for (int j = 0; j < V; j++) {
                        if (vocab[j] == c) inputs[i] = j;
                    }
                }
                for (int i = 0; i < n - 1; i++) {
                    targets[i] = inputs[i + 1];
                }

                // 順伝播 (Forward)
                double[] hPrev = new double[H];
                for (int i = 0; i < n - 1; i++) {
                    int xIdx = inputs[i];
                    double[] h = new double[H];
                    for (int j = 0; j < H; j++) {
                        double sum = bh[j];
                        sum += Wxh[j][xIdx];
                        for (int k = 0; k < H; k++) {
                            sum += Whh[j][k] * hPrev[k];
                        }
                        h[j] = Math.tanh(sum); // 活性化関数 tanh
                    }
                    hs[i] = h;

                    double[] logits = new double[V];
                    for (int j = 0; j < V; j++) {
                        double sum = by[j];
                        for (int k = 0; k < H; k++) {
                            sum += Why[j][k] * h[k];
                        }
                        logits[j] = sum;
                    }
                    ps[i] = softmax(logits);

                    hPrev = h;
                }

                // 逆伝播 (Backward / 時間を遡って勾配を計算)
                double[][] dWhy = new double[V][H];
                double[] dby = new double[V];
                double[][] dWhh = new double[H][H];
                double[][] dWxh = new double[H][V];
                double[] dbh = new double[H];

                double[] dhNext = new double[H];

                for (int i = n - 2; i >= 0; i--) {
                    double[] dy = new double[V];
                    for (int j = 0; j < V; j++) {
                        double targetVal = (j == targets[i]) ? 1.0 : 0.0;
                        dy[j] = ps[i][j] - targetVal;
                    }

                    for (int j = 0; j < V; j++) {
                        for (int k = 0; k < H; k++) {
                            dWhy[j][k] += dy[j] * hs[i][k];
                        }
                        dby[j] += dy[j];
                    }

                    double[] dh = new double[H];
                    for (int j = 0; j < H; j++) {
                        double sum = 0;
                        for (int k = 0; k < V; k++) {
                            sum += Why[k][j] * dy[k];
                        }
                        sum += dhNext[j];
                        dh[j] = sum;
                    }

                    double[] dtanh = new double[H];
                    for (int j = 0; j < H; j++) {
                        dtanh[j] = (1.0 - hs[i][j] * hs[i][j]) * dh[j];
                        dbh[j] += dtanh[j];
                    }

                    double[] hPrevStep = (i == 0) ? new double[H] : hs[i - 1];

                    for (int j = 0; j < H; j++) {
                        dWxh[j][inputs[i]] += dtanh[j];
                        for (int k = 0; k < H; k++) {
                            dWhh[j][k] += dtanh[j] * hPrevStep[k];
                        }
                    }

                    for (int j = 0; j < H; j++) {
                        double sum = 0;
                        for (int k = 0; k < H; k++) {
                            sum += Whh[k][j] * dtanh[k];
                        }
                        dhNext[j] = sum;
                    }
                }

                // パラメータの更新
                for (int j = 0; j < H; j++) {
                    bh[j] -= learningRate * dbh[j];
                    for (int k = 0; k < V; k++) {
                        Wxh[j][k] -= learningRate * dWxh[j][k];
                    }
                    for (int k = 0; k < H; k++) {
                        Whh[j][k] -= learningRate * dWhh[j][k];
                    }
                }
                for (int j = 0; j < V; j++) {
                    by[j] -= learningRate * dby[j];
                    for (int k = 0; k < H; k++) {
                        Why[j][k] -= learningRate * dWhy[j][k];
                    }
                }
            }
        }

        System.out.println("✨ RNNの学習が完了しました！\n");

        // 標準入力を受け付けてテキスト生成するループ
        Scanner scanner = new Scanner(System.in, StandardCharsets.UTF_8);
        while (true) {
            System.out.print("文字を入力してください: ");
            String input = scanner.nextLine().trim();

            if (input.equalsIgnoreCase("q")) {
                System.out.println("終了します");
                break;
            }

            if (input.isEmpty()) {
                System.out.println("文字を入力してください\n");
                continue;
            }

            // ボキャブラリーチェック
            boolean allValid = true;
            for (int i = 0; i < input.length(); i++) {
                boolean found = false;
                for (int j = 0; j < V; j++) {
                    if (vocab[j] == input.charAt(i)) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    allValid = false;
                    break;
                }
            }

            if (!allValid) {
                System.out.println("⚠️ ボキャブラリーに含まれない文字が含まれています。\n");
                continue;
            }

// 🌟 ユーザーの入力のあとに「$」を補って、AIに「ここまでが質問だよ」と教える
            String prompt = input + "$";

            double[] h = new double[H];
            for (int i = 0; i < prompt.length(); i++) {
                char c = prompt.charAt(i);
                int xIdx = -1;
                for (int j = 0; j < V; j++) {
                    if (vocab[j] == c) {
                        xIdx = j;
                        break;
                    }
                }
                if (xIdx == -1) continue;

                double[] nextH = new double[H];
                for (int j = 0; j < H; j++) {
                    double sum = bh[j];
                    sum += Wxh[j][xIdx];
                    for (int k = 0; k < H; k++) {
                        sum += Whh[j][k] * h[k];
                    }
                    nextH[j] = Math.tanh(sum);
                }
                h = nextH;
            }

            // 画面にはシンプルに「AI: 」から返事をスタートさせる
            System.out.print("AI: ");

            // 🌟 育った記憶（`h`）をベースに、続きの文字を自動生成していく
            for (int step = 0; step < mozisuuseigenn; step++) {
                double[] logits = new double[V];
                for (int j = 0; j < V; j++) {
                    double sum = by[j];
                    for (int k = 0; k < H; k++) {
                        sum += Why[j][k] * h[k];
                    }
                    logits[j] = sum;
                }
                double[] probs = softmax(logits);

                double r = rand.nextDouble(); // 0.0 から 1.0 の間のランダムな数字
                double cumulative = 0.0;
                int nextIdx = 0;
                for (int j = 0; j < V; j++) {
                    cumulative += probs[j];
                    if (r < cumulative) {
                        nextIdx = j;
                        break;
                    }
                }

                char nextChar = vocab[nextIdx];
                if (nextChar == '$') {
                    break;
                }

                System.out.print(nextChar);

                // 生成した文字を次の入力として扱い、隠れ状態 `h` をさらにアップデートする
                double[] nextH = new double[H];
                for (int j = 0; j < H; j++) {
                    double sum = bh[j];
                    sum += Wxh[j][nextIdx];
                    for (int k = 0; k < H; k++) {
                        sum += Whh[j][k] * h[k];
                    }
                    nextH[j] = Math.tanh(sum);
                }
                h = nextH;
            }
            System.out.println("\n");
        }
        scanner.close();
    }

    public static double[] softmax(double[] z) {
        double[] probs = new double[z.length];
        double max = z[0];
        for (double v : z) if (v > max) max = v;

        double sum = 0;
        for (int i = 0; i < z.length; i++) {
            probs[i] = Math.exp(z[i] - max);
            sum += probs[i];
        }
        for (int i = 0; i < z.length; i++) {
            probs[i] /= sum;
        }
        return probs;
    }
}