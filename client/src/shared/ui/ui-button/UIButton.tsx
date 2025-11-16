import { Pressable, PressableProps, StyleSheet, ViewStyle } from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { FC, ReactNode } from "react";
import UIText from "@/src/shared/ui/ui-text/UIText";
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withTiming,
} from "react-native-reanimated";
import { SvgProps } from "react-native-svg";

type ButtonVariant = "contained" | "outlined";
type ButtonColor = "emerald";

interface IProps extends PressableProps {
  children: ReactNode;
  variant?: ButtonVariant;
  color?: ButtonColor;
  leftIcon?: FC<SvgProps>;
}

const AnimatedPressable = Animated.createAnimatedComponent(Pressable);

export default function UIButton(props: IProps) {
  const {
    children,
    style: customStyles,
    variant = "contained",
    color = "emerald",
    leftIcon: LeftIcon,
    ...rest
  } = props;

  const isContained = variant === "contained";

  const scale = useSharedValue(1);
  const opacity = useSharedValue(1);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
    opacity: opacity.value,
  }));

  const onPressIn = () => {
    scale.value = withTiming(0.96, { duration: 120 });
    opacity.value = withTiming(0.85, { duration: 120 });
  };

  const onPressOut = () => {
    scale.value = withTiming(1, { duration: 120 });
    opacity.value = withTiming(1, { duration: 120 });
  };

  const gradientColors: Record<ButtonColor, [string, string]> = {
    emerald: ["#00d492", "#009966"] as const,
  };

  return (
    <AnimatedPressable
      {...rest}
      onPressIn={onPressIn}
      onPressOut={onPressOut}
      style={[animatedStyle, customStyles as ViewStyle]}
    >
      {isContained ? (
        <LinearGradient
          colors={gradientColors[color]}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={staticStyles.button}
        >
          {LeftIcon && <LeftIcon color={"#ffffff"} width={16} height={16} />}

          <UIText size={16} color="#ffffff" weight={400}>
            {children}
          </UIText>
        </LinearGradient>
      ) : (
        <UIText size={16}>{children}</UIText>
      )}
    </AnimatedPressable>
  );
}

const staticStyles = StyleSheet.create({
  button: {
    minHeight: 56,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 12,
    justifyContent: "center",
    alignItems: "center",
    flexDirection: "row",
    gap: 8,
  },
});
