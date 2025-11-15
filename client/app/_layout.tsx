import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import "@/src/shared/styles/global.css"
import {SafeAreaProvider, SafeAreaView} from "react-native-safe-area-context";

export default function RootLayout() {
  return (
		<SafeAreaProvider>
				<Stack>
						<Stack.Screen name="(tabs)" options={{ headerShown: false }} />
						<Stack.Screen name="modal" options={{ presentation: 'modal', title: 'Modal' }} />
				</Stack>
		</SafeAreaProvider>

  );
}
