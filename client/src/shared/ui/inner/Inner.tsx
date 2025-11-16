import {SafeAreaView} from "react-native-safe-area-context";
import {ReactNode} from "react";
import {StyleSheet} from "react-native";

interface IProps {
	children: ReactNode;
}

export default function Inner (props: IProps) {
	const {children} = props;
	return (
		<SafeAreaView style={staticStyles.container}>
			{children}
		</SafeAreaView>
	)
}

const staticStyles = StyleSheet.create({
	container: {
		paddingHorizontal: 16
	}
})