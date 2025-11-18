import React from "react";
import { StyleSheet, View } from "react-native";
import Inner from "@/src/shared/ui/inner/Inner";
import Card from "@/src/shared/ui/card/Card";
import UIText from "@/src/shared/ui/ui-text/UIText";
import UIButton from "@/src/shared/ui/ui-button/UIButton";
import ImagePicker from "react-native-image-crop-picker";
import CameraIcon from "@/src/shared/assets/svg/camera.svg";
import UploadIcon from "@/src/shared/assets/svg/upload.svg";
import AppIcon from "@/src/shared/assets/svg/app-icon.svg";
import CopyIcon from "@/src/shared/assets/svg/copy.svg";
import { LinearGradient } from "expo-linear-gradient";
import { useRouter } from "expo-router";

export default function HomeScreen() {
  const router = useRouter();

  const openCameraAndCrop = () => {
    ImagePicker.openCamera({
      cropping: true,
      freeStyleCropEnabled: true,
      includeBase64: true,
      cropperToolbarTitle: "Сфотографируйте текст",
    })
      .then((image) => {
        if (!image) return;

        router.push({
          pathname: "/loading",
          params: { imageUrl: image.path },
        });
      })
      .catch((error) => {
        console.log(error);
        router.push({
          pathname: "/error",
        });
      });
  };

  const openGalleryAndCrop = () => {
    ImagePicker.openPicker({
      cropping: true,
      freeStyleCropEnabled: true,
      includeBase64: true,
      mediaType: "photo",
      multiple: false, // выбрать одно фото
      cropperToolbarTitle: "Обрежьте фото",
    })
      .then((image) => {
        if (!image) return;

        router.push({
          pathname: "/loading",
          params: { imageUrl: image.path },
        });
      })
      .catch((error) => {
        console.log("Error opening gallery:", error);
      });
  };

  return (
    <Inner style={styles.inner}>
      <View style={styles.infoBlock}>
        <LinearGradient
          colors={["#00d492", "#009966"]}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.logo}
        >
          <AppIcon color="#ffffff" width={38} height={38} />
        </LinearGradient>
        <UIText size={14}>TextScan AI</UIText>
        <UIText style={styles.description} size={14}>
          Распознавание текста с фотографий с помощью искусственного интеллекта
        </UIText>
      </View>

      <Card style={styles.buttons}>
        <UIButton
          style={styles.button}
          leftIcon={CameraIcon}
          onPress={openCameraAndCrop}
        >
          Сделать фото
        </UIButton>

        <View style={styles.dividerBlock}>
          <View style={styles.dividerItem}></View>
          <UIText size={14}>или</UIText>
          <View style={styles.dividerItem}></View>
        </View>

        <UIButton
          style={styles.button}
          leftIcon={UploadIcon}
          variant="outlined"
          onPress={openGalleryAndCrop}
        >
          Загрузить из галереи
        </UIButton>
      </Card>

      <View style={styles.icons}>
        <View style={styles.iconBlock}>
          <View style={styles.iconImage}>
            <CameraIcon width={24} height={24} color={"#00a63e"} />
          </View>
          <UIText size={14}>Сфотографируйте</UIText>
        </View>
        <View style={styles.iconBlock}>
          <View style={styles.iconImage}>
            <AppIcon width={24} height={24} color={"#00a63e"} />
          </View>
          <UIText size={14}>Распознайте</UIText>
        </View>
        <View style={styles.iconBlock}>
          <View style={styles.iconImage}>
            <CopyIcon width={24} height={24} color={"#00a63e"} />
          </View>
          <UIText size={14}>Скопируйте</UIText>
        </View>
      </View>
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
  buttons: {
    display: "flex",
    flexDirection: "column",
    gap: 24,
    alignItems: "center",
  },
  button: {
    width: "100%",
  },
  dividerBlock: {
    display: "flex",
    flexDirection: "row",
    gap: 16,
    alignItems: "center",
  },
  dividerItem: {
    flexGrow: 1,
    height: 1,
    backgroundColor: "#e5e7eb",
  },
  infoBlock: {
    width: "100%",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 12,
  },
  logo: {
    width: 80,
    height: 80,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    borderRadius: 24,
  },
  inner: {
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    gap: 24,
  },
  description: {
    textAlign: "center",
    lineHeight: 20,
  },
  icons: {
    width: "100%",
    flexDirection: "row",
    justifyContent: "space-between",
  },
  iconBlock: {
    alignItems: "center",
    gap: 8,
  },
  iconImage: {
    width: 48,
    height: 48,
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#dcfce7",
    borderRadius: 100,
  },
});
