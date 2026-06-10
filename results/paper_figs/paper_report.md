# ARTPS Makale Figür Özeti

- Toplam örnek: 5
- Curiosity ort/Std: 0.707 / 0.108
- AE MSE ort/Std: 0.008343 / 0.004375
- Depth variance ort/Std: 0.083996 / 0.005591
- Roughness ort/Std: 0.036435 / 0.020033

## Dağılım ve İlişkiler

![Dataset Summary](dataset_summary.png)

![Korelasyon Isı Haritası](corr_heatmap.png)

## En İlginç Örnekler (Top-5)

![Top Grid](topk_grid.png)

## En Düşük Curiosity (Bottom-5)

![Bottom Grid](bottomk_grid.png)

## Anomali Tespit Örnekleri (Kutulu Overlay)

Bu bölümde, her görsel için birleşik anomali haritası üzerinden üretilen kutulu tespit overlay örnekleri verilmektedir. (Yöntem: AE farkı + derinlik kenarı + doku/gölge + PaDiM/PatchCore füzyonu.) Yaklaşım, gezgin otonomisinde hedef önceliklendirmeye yönelik literatürle uyumludur [Estlin et al., 2014] (bkz. [JPL 2014 ISAIRAS](https://ai.jpl.nasa.gov/public/documents/papers/estlin-isairas2014-automated.pdf)).

![0735MR0031500040403079E01_DXXX_det_overlay.png](detection_overlays/0735MR0031500040403079E01_DXXX_det_overlay.png)

![FRF_0940_0750382098_770ECM_N0460000FHAZ00206_01_295J_calib01_areo.info_det_overlay.png](detection_overlays/FRF_0940_0750382098_770ECM_N0460000FHAZ00206_01_295J_calib01_areo.info_det_overlay.png)

![curiosity_0000_Sol_958__Mast_Camera_(Mastcam)_det_overlay.png](detection_overlays/curiosity_0000_Sol_958__Mast_Camera_(Mastcam)_det_overlay.png)

![curiosity_1100_MAST_1460_jpg.rf.f546b807109c1df632cb62e069ded089_det_overlay.png](detection_overlays/curiosity_1100_MAST_1460_jpg.rf.f546b807109c1df632cb62e069ded089_det_overlay.png)

![curiosity_1100_NAVCAM_540_jpg.rf.6315c37bd960ce862e4c6161408009cf_det_overlay.png](detection_overlays/curiosity_1100_NAVCAM_540_jpg.rf.6315c37bd960ce862e4c6161408009cf_det_overlay.png)
