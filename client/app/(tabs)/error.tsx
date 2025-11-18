import Inner from "@/src/shared/ui/inner/Inner";
import UIText from "@/src/shared/ui/ui-text/UIText";
import { StyleSheet } from "react-native";
import UIButton from "@/src/shared/ui/ui-button/UIButton";
import React from "react";
import { useRouter } from "expo-router";

export default function ErrorScreen() {
  const router = useRouter();
  const backHome = () => {
    router.replace("/");
  };
  return (
    <Inner style={styles.inner}>
      <UIText size={16}>Произошла ошибка. Попробуйте заново</UIText>
      <UIButton style={styles.button} onPress={backHome}>
        На главную
      </UIButton>
    </Inner>
  );
}

const styles = StyleSheet.create({
  inner: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    gap: 24,
  },
  button: {
    width: "60%",
  },
});
