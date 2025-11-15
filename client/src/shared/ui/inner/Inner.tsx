import {SafeAreaView} from "react-native-safe-area-context";
import {ReactNode} from "react";

interface IProps {
	children: ReactNode;
}

export default function Inner (props: IProps) {
	const {children} = props;
	return (
		<SafeAreaView>
			{children}
		</SafeAreaView>
	)
}