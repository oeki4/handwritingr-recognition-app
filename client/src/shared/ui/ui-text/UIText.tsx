import { ReactNode } from "react";
import {
  StyleSheet,
  Text as RNText,
  TextProps as RNTextProps,
} from "react-native";

type TextWeight = 400 | 500 | 600 | 700;

interface IProps extends RNTextProps {
  children?: ReactNode;
  size?: number;
  color?: string;
  weight?: TextWeight;
}

export default function UIText(props: IProps) {
  const { children, size = 12, color, weight, style } = props;
  return (
    <RNText
      style={StyleSheet.compose(
        {
          fontSize: size,
          flexShrink: 1,
          color,
          fontWeight: weight,
        },
        style,
      )}
    >
      {children}
    </RNText>
  );
}
