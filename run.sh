python3 main.py \
--image_dir "data/MVTec_LOCO" \
--cls 0 \
--subcls 1 \
--begin_of_image 0 \
--end_of_image 0 \
--begin_of_function 1 \
--end_of_function 4 \
--output_dir "result" \

# --cls 0:breakfast box, 1:juice_bottle, 2:pushpins, 3:screw_bag, 4:splicing_connectors
# --subcls 0:good(normal), 1:Logical_anomalies, 2:Structural_anomalies
# --begin_of_image(000.png, 001.png, 002.png, ...): the first image to be processed
# --end_of_image(000.png, 001.png, 002.png, ...): the last image to be processed
# --begin_of_function(0, 1, 2, ...): the first function to be processed
# --end_of_function(0, 1, 2, ...): the last function to be processed