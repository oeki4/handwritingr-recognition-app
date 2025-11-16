import {StyleSheet, View, Text} from "react-native";
import {ReactNode} from "react";

interface IProps {
	children?: ReactNode;
}

export default function Card(props: IProps) {
	const {children} = props;
	return (
		<View style={staticStyles.card}>
			{children}
		</View>
	)
}

const staticStyles = StyleSheet.create({
	card: {
		backgroundColor: "#FFFFFE",
		elevation: 12,
		shadowColor: 'rgba(0,0,0,0.1)',
		padding: 32,
		borderRadius: 10,
	}
})