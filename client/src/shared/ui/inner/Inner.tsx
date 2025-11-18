import { SafeAreaView } from "react-native-safe-area-context";
import { ReactNode } from "react";
import { StyleSheet, ViewStyle } from "react-native";
import { StatusBar } from "expo-status-bar";

interface IProps {
  children: ReactNode;
  style?: ViewStyle;
}

export default function Inner(props: IProps) {
  const { children, style } = props;
  return (
    <SafeAreaView style={StyleSheet.compose(staticStyles.container, style)}>
      <StatusBar style="dark" />
      {children}
    </SafeAreaView>
  );
}

const staticStyles = StyleSheet.create({
  container: {
    paddingHorizontal: 16,
    flex: 1,
  },
});
