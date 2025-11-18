import Inner from "@/src/shared/ui/inner/Inner";
import UIText from "@/src/shared/ui/ui-text/UIText";
import Card from "@/src/shared/ui/card/Card";
import { Image, Share, StyleSheet, View } from "react-native";
import { Link, useLocalSearchParams } from "expo-router";
import CheckmarkIcon from "@/src/shared/assets/svg/checkmark.svg";
import CopyIcon from "@/src/shared/assets/svg/copy.svg";
import ShareIcon from "@/src/shared/assets/svg/share.svg";
import UIButton from "@/src/shared/ui/ui-button/UIButton";
import React from "react";
import * as Clipboard from "expo-clipboard";

export default function ResultScreen() {
  const { imageUrl, result } = useLocalSearchParams();
  const parsedResult = JSON.parse(result);

  const handleCopy = async () => {
    try {
      await Clipboard.setStringAsync(parsedResult.full_text);
      console.log("Текст скопирован");
    } catch (e) {
      console.log("Ошибка копирования:", e);
    }
  };

  const handleShare = async () => {
    try {
      await Share.share({
        message: parsedResult.full_text,
      });
    } catch (e) {
      console.log("Ошибка отправки:", e);
    }
  };

  return (
    <Inner style={styles.inner}>
      <View style={styles.flexRow}>
        <UIText size={14}>Результат</UIText>

        <Link href="/">
          <UIText color={"#008236"} size={14}>
            Новое сканирование
          </UIText>
        </Link>
      </View>

      <Card>
        <Image
          source={{ uri: String(imageUrl) }}
          style={styles.imageBox}
          resizeMode="stretch"
        />
      </Card>

      <Card style={styles.recognizedTextBlock}>
        <View style={styles.recognizedTextHeader}>
          <UIText size={14}>Распознанный текст</UIText>
          <View style={styles.badge}>
            <CheckmarkIcon color={"#008236"} width={16} height={16} />
            <UIText color={"#008236"} size={14}>
              Готово
            </UIText>
          </View>
        </View>

        <View style={styles.recognizedText}>
          <UIText>{parsedResult.full_text}</UIText>
        </View>

        <View style={styles.flexRow}>
          <UIButton
            leftIcon={CopyIcon}
            style={styles.button}
            onPress={handleCopy}
          >
            Копировать
          </UIButton>

          <UIButton
            leftIcon={ShareIcon}
            variant={"outlined"}
            style={styles.button}
            onPress={handleShare}
          >
            Поделиться
          </UIButton>
        </View>
      </Card>
    </Inner>
  );
}

const styles = StyleSheet.create({
  inner: {
    display: "flex",
    flexDirection: "column",
    gap: 16,
    justifyContent: "center",
  },
  flexRow: {
    display: "flex",
    justifyContent: "space-between",
    flexDirection: "row",
    alignItems: "center",
    width: "100%",
    gap: 16,
  },
  imageBox: {
    height: 300,
    overflow: "hidden",
    borderRadius: 14,
  },
  recognizedText: {
    borderWidth: 1,
    borderStyle: "solid",
    borderColor: "#e5e7eb",
    width: "100%",
    padding: 16,
    borderRadius: 14,
  },
  recognizedTextBlock: {
    display: "flex",
    flexDirection: "column",
    gap: 24,
  },
  recognizedTextHeader: {
    display: "flex",
    width: "100%",
    flexDirection: "row",
    justifyContent: "space-between",
  },
  badge: {
    paddingHorizontal: 12,
    paddingVertical: 4,
    display: "flex",
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    backgroundColor: "#dcfce7",
    borderRadius: 9999,
  },
  button: {
    flexGrow: 1,
  },
});
