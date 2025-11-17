import { StyleSheet, View, ViewStyle } from "react-native";
import { ReactNode } from "react";

interface IProps {
  children?: ReactNode;
  style?: ViewStyle;
}

export default function Card(props: IProps) {
  const { children, style } = props;
  return (
    <View style={StyleSheet.compose(staticStyles.card, style)}>{children}</View>
  );
}

const staticStyles = StyleSheet.create({
  card: {
    backgroundColor: "#FFFFFE",
    elevation: 12,
    shadowColor: "rgba(0,0,0,0.1)",
    padding: 32,
    borderRadius: 10,
  },
});
