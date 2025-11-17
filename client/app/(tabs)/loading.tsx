import React from "react";
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
import { useLocalSearchParams } from "expo-router";
import { LinearGradient } from "expo-linear-gradient";
import UIText from "@/src/shared/ui/ui-text/UIText";
import AppIcon from "@/src/shared/assets/svg/app-icon.svg";

export default function LoadingScanScreen() {
  const { imageUrl } = useLocalSearchParams();

  const translateY = useSharedValue(0);
  const scanHeight = useSharedValue(0);
  const pulseOpacity = useSharedValue(0.3);
  const rotation = useSharedValue(0); // Добавляем значение для вращения

  const LINE_HEIGHT = 3;
  const DURATION = 1000;
  const PULSE_DURATION = 1000;
  const ROTATION_DURATION = 2000; // Длительность полного оборота

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ translateY: translateY.value }],
  }));

  const gradientAnimatedStyle = useAnimatedStyle(() => {
    return {
      opacity: pulseOpacity.value,
    };
  });

  // Анимированный стиль для вращения иконки
  const iconAnimatedStyle = useAnimatedStyle(() => {
    return {
      transform: [{ rotate: `${rotation.value}deg` }],
    };
  });

  // запуск анимации после измерения высоты
  const onImageLayout = (e) => {
    const h = e.nativeEvent.layout.height;
    // сохраняем высоту (можно использовать, если нужно)
    scanHeight.value = h;

    // стартуем линию чуть выше, чтобы сначала шла сверху
    translateY.value = -LINE_HEIGHT;

    // двигаем до нижней границы (h - LINE_HEIGHT), затем обратно, бесконечно
    translateY.value = withRepeat(
      withTiming(Math.max(0, h - LINE_HEIGHT), {
        duration: DURATION,
        easing: Easing.linear,
      }),
      -1, // infinite
      true, // reverse
    );

    // запускаем пульсацию градиента
    pulseOpacity.value = withRepeat(
      withTiming(0.5, {
        duration: PULSE_DURATION,
        easing: Easing.inOut(Easing.ease),
      }),
      -1, // infinite
      true, // reverse
    );

    // запускаем вращение иконки
    rotation.value = withRepeat(
      withTiming(360, {
        duration: ROTATION_DURATION,
        easing: Easing.linear,
      }),
      -1, // infinite
      false, // не реверсировать - сразу начинать новый цикл
    );
  };

  return (
    <Inner style={styles.inner}>
      <Card>
        <View style={styles.imageWrapper} onLayout={onImageLayout}>
          {/* Если imageUrl не передан — можно показать плейсхолдер */}
          <Image
            source={{ uri: String(imageUrl) }}
            style={styles.image}
            resizeMode="cover"
          />

          {/* Мигающий зеленый градиент */}
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
          <UIText
            style={{
              textAlign: "center",
            }}
            size={14}
          >
            Анализ изображения
          </UIText>
          <UIText
            style={{
              textAlign: "center",
            }}
            size={14}
          >
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
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
  },
});
