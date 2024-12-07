from functions import image_segmentation_generator


def train(model,
          train_images,
          train_annotations,
          epochs=5,
          batch_size=2,
          steps_per_epoch=512,
          validate=True,  
          val_images=None,
          val_annotations=None,
          val_steps_per_epoch=512,
          optimizer_name='adam',
          do_augment=False,
          input_height=None,
          input_width=None,
          n_classes=None,
          preprocessing=None,
          read_image_type=1  
         ):

    # Get model dimensions and number of classes
    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width

    # Ensure validation data is provided when `validate` is True
    if validate:
        assert val_images is not None
        assert val_annotations is not None

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer_name,
                  metrics=['accuracy'])

    
    train_gen = image_segmentation_generator(
        train_images, train_annotations, batch_size, n_classes,
        input_height, input_width, output_height, output_width,
        do_augment=do_augment, preprocessing=preprocessing, read_image_type=read_image_type)
    
    val_gen = image_segmentation_generator(
        val_images, val_annotations, batch_size, n_classes,
        input_height, input_width, output_height, output_width,
        preprocessing=preprocessing, read_image_type=read_image_type)
    #debugging
    data, labels = next(train_gen)
    print("Data batch shape:", data.shape)
    print("Labels batch shape:", labels.shape)
    
    model.fit(train_gen, steps_per_epoch=steps_per_epoch,
            validation_data=val_gen, validation_steps=val_steps_per_epoch,
            epochs=epochs)
