import React, { useState } from "react";
import { Image, StyleSheet, View } from "react-native";
import Inner from "@/src/shared/ui/inner/Inner";
import Card from "@/src/shared/ui/card/Card";
import UIText from "@/src/shared/ui/ui-text/UIText";
import UIButton from "@/src/shared/ui/ui-button/UIButton";
import ImagePicker from "react-native-image-crop-picker";

export default function HomeScreen() {
  const [image, setImage] = useState(null);

  const openCameraAndCrop = () => {
    ImagePicker.openCamera({
      width: 300,
      height: 400,
      cropping: true,
      freeStyleCropEnabled: true,
      includeBase64: true,
    })
      .then((image) => {
        setImage(image.path);
      })
      .catch((error) => {
        console.log("Error opening camera:", error);
      });
  };

  return (
    <Inner>
      <Card>
        <UIText size={14}>Сделай фото и обрежь как хочешь</UIText>
        <UIButton onPress={openCameraAndCrop}>Открыть камеру</UIButton>
        {image && (
          <View style={styles.imageContainer}>
            <Image source={{ uri: image }} style={styles.image} />
          </View>
        )}
      </Card>
    </Inner>
  );
}

const styles = StyleSheet.create({
  imageContainer: {
    marginTop: 20,
    alignItems: "center",
  },
  image: {
    width: 300,
    height: 400,
    borderRadius: 10,
  },
});
