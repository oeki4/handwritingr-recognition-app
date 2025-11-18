import { Stack } from "expo-router";

import { SafeAreaProvider } from "react-native-safe-area-context";
import "react-native-reanimated";

export default function RootLayout() {
  return (
    <SafeAreaProvider>
      <Stack>
        <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
        <Stack.Screen
          name="modal"
          options={{ presentation: "modal", title: "Modal" }}
        />
      </Stack>
    </SafeAreaProvider>
  );
}
