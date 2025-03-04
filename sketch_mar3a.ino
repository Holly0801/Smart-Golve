void setup() {
  Serial.begin(9600);  // 初始化串口通信
}

void loop() {
  for (int i = 0; i < 5; i++) {
    int sensorValue = analogRead(A0 + i);  // 读取A0-A4的传感器数据
    Serial.print(sensorValue);
    if (i < 4) {
      Serial.print(",");  // 用逗号分隔数据
    }
  }
  Serial.println();  // 换行
  delay(100);        // 适当延迟
}
