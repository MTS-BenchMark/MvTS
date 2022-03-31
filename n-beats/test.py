import numpy as np

#from nbeats_keras.model import NBeatsNet as NBeatsKeras
from nbeats_pytorch.model import NBeatsNet as NBeatsPytorch

#warnings.filterwarnings(action='ignore', message='Setting attributes')


def main():
    # https://keras.io/layers/recurrent/
    # At the moment only Keras supports input_dim > 1. In the original paper, input_dim=1.
    num_samples, time_steps, input_dim, output_dim = 50_000, 12, 1, 1

    # This example is for both Keras and Pytorch. In practice, choose the one you prefer. NBeatsKeras,
    for BackendType in [NBeatsPytorch]:
        # NOTE: If you choose the Keras backend with input_dim>1, you have 
        # to set the value here too (in the constructor).
        backend = BackendType(
            backcast_length=time_steps, forecast_length=12,
            stack_types=(NBeatsPytorch.GENERIC_BLOCK, NBeatsPytorch.GENERIC_BLOCK),
            nb_blocks_per_stack=2, thetas_dim=(4, 4), share_weights_in_stack=True,
            hidden_layer_units=64
        )

        # Definition of the objective function and the optimizer.
        backend.compile(loss='mae', optimizer='adam')

        # Definition of the data. The problem to solve is to find f such as | f(x) - y | -> 0.
        # where f = np.mean.
        x = np.random.uniform(size=(num_samples, time_steps, input_dim))
        y = np.mean(x, axis=1, keepdims=True)

        # Split data into training and testing datasets.
        c = num_samples // 10
        x_train, y_train, x_test, y_test = x[c:], y[c:], x[:c], y[:c]
        test_size = len(x_test)

        # Train the model.
        print('Training...')
        backend.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=128)

        # Save the model for later.
        backend.save('n_beats_model.h5')

        # Predict on the testing set (forecast).
        predictions_forecast = backend.predict(x_test)
        print(predictions_forecast.shape)
        #np.testing.assert_equal(predictions_forecast.shape, (test_size, backend.forecast_length, output_dim))

        # Predict on the testing set (backcast).
        #predictions_backcast = backend.predict(x_test, return_backcast=True)
        #np.testing.assert_equal(predictions_backcast.shape, (test_size, backend.backcast_length, output_dim))

        # Load the model.
        #model_2 = BackendType.load('n_beats_model.h5')

        #np.testing.assert_almost_equal(predictions_forecast, model_2.predict(x_test))


if __name__ == '__main__':
    main()