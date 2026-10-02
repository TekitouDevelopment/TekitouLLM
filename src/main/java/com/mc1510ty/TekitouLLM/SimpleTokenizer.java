package com.mc1510ty.TekitouLLM;

import java.util.*;

public class SimpleTokenizer {
    private Map<String, Integer> tokenToId = new HashMap<>();
    private Map<Integer, String> idToToken = new HashMap<>();
    private List<String> merges = new ArrayList<>(); // マージした履歴（順番が重要！）

    private static final List<String> SPECIAL_TOKENS = Arrays.asList("U:", ">A:", "■");

    public void train(String[] corpus, int targetVocabSize, int minFrequency) {
        int id = 0;
        tokenToId.clear();
        idToToken.clear();
        merges.clear();

        // 1. 特殊トークンを辞書に登録
        for (String st : SPECIAL_TOKENS) {
            tokenToId.put(st, id);
            idToToken.put(id, st);
            id++;
        }

        // 2. 特殊トークン以外の文字をベースとして登録
        Set<Character> uniqueChars = new HashSet<>();
        for (String text : corpus) {
            String temp = text;
            for (String st : SPECIAL_TOKENS) {
                temp = temp.replace(st, "");
            }
            for (char c : temp.toCharArray()) {
                uniqueChars.add(c);
            }
        }

        for (char c : uniqueChars) {
            String s = String.valueOf(c);
            if (!tokenToId.containsKey(s)) {
                tokenToId.put(s, id);
                idToToken.put(id, s);
                id++;
            }
        }

        // 3. コーパスを初期トークンリストに分解
        List<List<String>> splitCorpus = new ArrayList<>();
        for (String text : corpus) {
            splitCorpus.add(tokenizeInitial(text));
        }

        // 4. BPEマージの実行
        while (tokenToId.size() < targetVocabSize) {
            Map<String, Integer> pairCounts = new HashMap<>();

            for (List<String> tokens : splitCorpus) {
                for (int i = 0; i < tokens.size() - 1; i++) {
                    String t1 = tokens.get(i);
                    String t2 = tokens.get(i + 1);

                    if (SPECIAL_TOKENS.contains(t1) || SPECIAL_TOKENS.contains(t2)) {
                        continue;
                    }

                    String pair = t1 + "\u0001" + t2;
                    pairCounts.put(pair, pairCounts.getOrDefault(pair, 0) + 1);
                }
            }

            if (pairCounts.isEmpty()) break;

            String bestPair = null;
            int maxCount = -1;
            for (Map.Entry<String, Integer> entry : pairCounts.entrySet()) {
                if (entry.getValue() > maxCount) {
                    maxCount = entry.getValue();
                    bestPair = entry.getKey();
                }
            }

            if (bestPair == null || maxCount < minFrequency) {
                break;
            }

            String[] parts = bestPair.split("\u0001");
            String merged = parts[0] + parts[1];
            merges.add(bestPair); // どの順番でマージしたかを記録！

            if (!tokenToId.containsKey(merged)) {
                tokenToId.put(merged, id);
                idToToken.put(id, merged);
                id++;
            } else {
                break;
            }

            for (List<String> tokens : splitCorpus) {
                for (int i = 0; i < tokens.size() - 1; i++) {
                    if (tokens.get(i).equals(parts[0]) && tokens.get(i + 1).equals(parts[1])) {
                        tokens.set(i, merged);
                        tokens.remove(i + 1);
                    }
                }
            }
        }

        System.out.println("トークナイザーの学習完了！ 最終語彙数: " + tokenToId.size());
    }

    private List<String> tokenizeInitial(String text) {
        List<String> result = new ArrayList<>();
        int i = 0;
        while (i < text.length()) {
            boolean matched = false;
            for (String st : SPECIAL_TOKENS) {
                if (text.startsWith(st, i)) {
                    result.add(st);
                    i += st.length();
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                result.add(String.valueOf(text.charAt(i)));
                i++;
            }
        }
        return result;
    }

    // ★【本格実装】学習済みの merges ルールを適用して正しくサブワードに分割する
    public List<Integer> encode(String text) {
        List<String> tokens = tokenizeInitial(text);

        // 学習時に記録したマージルールを順番に適用していく
        for (String merge : merges) {
            String[] parts = merge.split("\u0001");
            String p1 = parts[0];
            String p2 = parts[1];
            String merged = p1 + p2;

            List<String> newTokens = new ArrayList<>();
            for (int i = 0; i < tokens.size(); i++) {
                if (i < tokens.size() - 1 && tokens.get(i).equals(p1) && tokens.get(i + 1).equals(p2)) {
                    newTokens.add(merged);
                    i++; // 次の要素をスキップ
                } else {
                    newTokens.add(tokens.get(i));
                }
            }
            tokens = newTokens;
        }

        // トークン文字列をIDに変換
        List<Integer> result = new ArrayList<>();
        for (String t : tokens) {
            result.add(tokenToId.getOrDefault(t, 0)); // 辞書になければ0にフォールバック
        }
        return result;
    }

    public String decode(List<Integer> ids) {
        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            sb.append(idToToken.getOrDefault(id, ""));
        }
        return sb.toString();
    }

    public int getVocabSize() {
        return tokenToId.size();
    }

    public String decodeToken(int id) {
        return idToToken.getOrDefault(id, "");
    }

}