package com.mc1510ty.TekitouLLM;

public class Main {
    public static void main(String[] args) {

//        double[] inputs  = {1.0, 2.0, 1.0, 2.0};
//        double[] targets = {2.0, 1.0, 2.0, 1.0};

                double[] inputs  = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0};
        double[] targets = {2.0, 3.0, 1.0, 2.0, 3.0, 1.0};

//        double[] inputs  = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0};
//        double[] targets = {2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0};

        // 重みとバイアスの初期値（新たに「記憶用の重み: wMemory」を追加！）
        double wInput = 0.5;
        double wMemory = 0.3;
        double b = 0.0;
        double learningRate = 0.05;

        System.out.println("=== 記憶（コンテキスト）を組み込んだ学習コード ===");

        for (int epoch = 1; epoch <= 100000; epoch++) {
            double totalError = 0.0;
            double memory = 0.0; // 1周のはじめは、記憶をまっさら（ゼロ）にする

            for (int i = 0; i < inputs.length; i++) {
                double x = inputs[i];
                double t = targets[i];

                // ① 予測する（「今の入力」と「前回の記憶」を両方使う！）
                double pred = (wInput * x) + (wMemory * memory) + b;

                // ② 誤差（ズレ）を計算する
                double error = pred - t;
                totalError += Math.abs(error);

                // ③ 重みとバイアス、そして記憶の重みを修正する
                wInput  -= learningRate * error * x;
                wMemory -= learningRate * error * memory;
                b       -= learningRate * error;

                // ④ 次のステップのために、今回の入力を「次の記憶」としてバトンタッチする
                memory = x;
            }

            // 150周ごとに誤差を表示
            if (epoch % 150 == 0 || epoch == 1) {
                System.out.printf("第 %3d 周目 | 誤差の合計: %.4f\n", epoch, totalError);
            }
        }

        System.out.println("\n--- テストしてみる ---");

        // テスト：「直前の記憶が 2.0」で、今回「1.0」が入力されたとき、次はいくつと予測する？
        double testMemory = 1.0;
        double testInput = 2.0;
        double result = (wInput * testInput) + (wMemory * testMemory) + b;

        System.out.printf("直前の記憶「%.1f」、今回の入力「%.1f」の次に来る予測値: %.4f\n",
                testMemory, testInput, result);

        IO.println("wInput: " + wInput + " wMemory: " + wMemory+ " b: " + b);

    }
}