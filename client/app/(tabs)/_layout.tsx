import { Tabs } from "expo-router";
import React from "react";

export default function TabLayout() {
  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarStyle: { display: "none" },
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          title: "Home",
        }}
      />
      <Tabs.Screen
        name="loading"
        options={{
          title: "Loading",
        }}
      />
      <Tabs.Screen
        name="result"
        options={{
          title: "Result",
        }}
      />
    </Tabs>
  );
}
