import React, { useEffect } from "react";
import { Image, StyleSheet, View } from "react-native";
import Animated, {
  Easing,
  useAnimatedStyle,
  useSharedValue,
  withRepeat,
  withTiming,
} from "react-native-reanimated";
import Inner from "@/src/shared/ui/inner/Inner";
import Card from "@/src/shared/ui/card/Card";
import { useLocalSearchParams, useRouter } from "expo-router";
import { LinearGradient } from "expo-linear-gradient";
import UIText from "@/src/shared/ui/ui-text/UIText";
import AppIcon from "@/src/shared/assets/svg/app-icon.svg";
import * as FileSystem from "expo-file-system/legacy";

import axios from "axios";
import { API_URL } from "@/src/shared/config/config";

export default function LoadingScanScreen() {
  const { imageUrl } = useLocalSearchParams();
  const router = useRouter();

  const translateY = useSharedValue(0);
  const scanHeight = useSharedValue(0);
  const pulseOpacity = useSharedValue(0.3);
  const rotation = useSharedValue(0);

  const LINE_HEIGHT = 3;
  const DURATION = 1000;
  const PULSE_DURATION = 1000;
  const ROTATION_DURATION = 2000;

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ translateY: translateY.value }],
  }));

  const gradientAnimatedStyle = useAnimatedStyle(() => ({
    opacity: pulseOpacity.value,
  }));

  const iconAnimatedStyle = useAnimatedStyle(() => ({
    transform: [{ rotate: `${rotation.value}deg` }],
  }));

  // ---------- ЗАПРОС НА backend ----------

  const sendImage = async () => {
    try {
      const fileUri = String(imageUrl);

      const fileInfo = await FileSystem.getInfoAsync(fileUri);
      if (!fileInfo.exists) {
        console.warn("Файл не найден по пути:", fileUri);
        return;
      }

      const formData = new FormData();
      formData.append("file", {
        uri: fileUri,
        type: "image/jpeg",
        name: "uploaded.jpg",
      });

      const response = await axios.post(
        `${API_URL}/recognize?return_image=true`,
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        },
      );

      console.log("Ответ сервера:", response.data);

      router.push({
        pathname: "/result",
        params: {
          imageUrl: fileUri,
          result: JSON.stringify(response.data),
        },
      });
    } catch (error) {
      console.error("Ошибка загрузки:", error);
    }
  };

  // --------- Старт при загрузке ----------
  useEffect(() => {
    sendImage();
  }, [imageUrl]);

  // Анимации
  const onImageLayout = (e) => {
    const h = e.nativeEvent.layout.height;
    scanHeight.value = h;

    translateY.value = -LINE_HEIGHT;

    translateY.value = withRepeat(
      withTiming(Math.max(0, h - LINE_HEIGHT), {
        duration: DURATION,
        easing: Easing.linear,
      }),
      -1,
      true,
    );

    pulseOpacity.value = withRepeat(
      withTiming(0.5, {
        duration: PULSE_DURATION,
        easing: Easing.inOut(Easing.ease),
      }),
      -1,
      true,
    );

    rotation.value = withRepeat(
      withTiming(360, {
        duration: ROTATION_DURATION,
        easing: Easing.linear,
      }),
      -1,
      false,
    );
  };

  return (
    <Inner style={styles.inner}>
      <Card>
        <View style={styles.imageWrapper} onLayout={onImageLayout}>
          <Image
            source={{ uri: String(imageUrl) }}
            style={styles.image}
            resizeMode="stretch"
          />

          <Animated.View
            style={[styles.gradientOverlay, gradientAnimatedStyle]}
          >
            <LinearGradient
              colors={[
                "rgba(0, 255, 0, 0.1)",
                "rgba(0, 255, 100, 0.2)",
                "rgba(0, 200, 0, 0.1)",
              ]}
              style={styles.gradient}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 1 }}
            />
          </Animated.View>

          <Animated.View style={[styles.scanLine, animatedStyle]} />
        </View>

        <View style={styles.description}>
          <Animated.View style={iconAnimatedStyle}>
            <AppIcon width={50} color={"#00a63e"} height={50} />
          </Animated.View>

          <UIText style={{ textAlign: "center" }} size={14}>
            Анализ изображения
          </UIText>

          <UIText style={{ textAlign: "center" }} size={14}>
            Модель машинного обучения распознает текст...
          </UIText>
        </View>
      </Card>
    </Inner>
  );
}

const styles = StyleSheet.create({
  imageWrapper: {
    width: "100%",
    height: 300,
    overflow: "hidden",
    position: "relative",
    borderRadius: 10,
  },
  image: {
    width: "100%",
    height: "100%",
    borderRadius: 10,
  },
  scanLine: {
    position: "absolute",
    top: 0,
    left: 0,
    width: "100%",
    height: 3,
    backgroundColor: "white",
    opacity: 0.9,
    zIndex: 2,
  },
  gradientOverlay: {
    position: "absolute",
    top: 0,
    left: 0,
    width: "100%",
    height: "100%",
    zIndex: 1,
    backgroundColor: "white",
  },
  gradient: {
    width: "100%",
    height: "100%",
    borderRadius: 10,
  },
  description: {
    marginTop: 24,
    display: "flex",
    flexDirection: "column",
    width: "100%",
    alignItems: "center",
    gap: 12,
  },
  inner: {
    flexDirection: "column",
    justifyContent: "center",
  },
});
