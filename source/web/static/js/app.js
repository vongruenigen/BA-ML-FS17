$(function () {
  // URL definitions
  var getModelsPath = '/get_models',
      getSessionPath = '/get_session',
      startSessionPath = '/start_session/<model>',
      stopSessionPath = '/stop_session',
      runInferencePath = '/run_inference';

  // DOM objects
  var $modelDropdown = $('#model_entries'),
      $startButton = $('#start_button'),
      $stopButton = $('#stop_button'),
      $inputText = $('#input_text'),
      $modelAnswers = $('#model_answers'),
      $outterContainer = $('#content_outter_container'),
      $loadingContainer = $('#loading_container'),
      $sendButton = $('#send_button'),
      $modelAnswers = $('#model_answers');

  // Other variables
  var currentSession = null;

  // Helper functions
  var enableStartControls = function (enable) {
    if (enable) {
      $modelDropdown.removeAttr('disabled');
      $startButton.removeAttr('disabled');
      $stopButton.attr('disabled', 'disabled');
    } else {
      $modelDropdown.attr('disabled', 'disabled');
      $startButton.attr('disabled', 'disabled');
      $stopButton.removeAttr('disabled');
    }
  };

  // Build models drodown on page load
  $.get(getModelsPath, function (models) {
    models.forEach(function (m) {
      var $modelOption = $('<option />');
      $modelOption.val(m)
      $modelOption.text(m);
      $modelOption.appendTo($modelDropdown);
    });

    // Disable/enable buttons based on current session
    $.get(getSessionPath, function (currentSession) {
      if (currentSession) {
        $modelDropdown.val(currentSession);
        enableStartControls(false);
        $outterContainer.show();
      }
    }).fail(function () {
      enableStartControls(true);
    });
  });

  // Start session when the user clicks on the button
  $startButton.click(function () {
    var currSeletedModel = $modelDropdown.val();

    if (!currSeletedModel) {
      alert('Please select a model!');
      return;
    }

    var urlCurrSelectedModel = startSessionPath.replace('<model>', encodeURI(currSeletedModel));

    $loadingContainer.show();

    $.post(urlCurrSelectedModel, function () {
      $loadingContainer.hide();
      $outterContainer.show();
      enableStartControls(false); 
    }).fail(function (err) {
      alert('Error while starting the session: ' + err.responseText);
      $loadingContainer.hide();
      enableStartControls(true);
    });
  });

  // Stop the session when the user clicks on the button
  $stopButton.click(function () {
    var currSeletedModel = $modelDropdown.val();

    if (!currSeletedModel) {
      alert('No model currently selected!');
      return;
    }

    $.post(stopSessionPath, function () {
      $outterContainer.hide();
      enableStartControls(true);
      alert('Session stopped.');
    }).fail(function (err) {
      alert('Error while stopping the session: ' + err.responseText);
      enableStartControls(false);
    });
  });

  $inputText.keypress(function (e) {
    if (e.which === 13) {
      $sendButton.trigger('click');
    }
  });

  $sendButton.click(function () {
    var newText = $inputText.val();

    $.post(runInferencePath, newText, function (responseText) {
      newEntry = 'Input:\t' + newText + '\n';
      newEntry += 'Answer:\t' + responseText + '\n\n';
      $modelAnswers.val($modelAnswers.val() + newEntry);
    }).fail(function (err) {
      alert('Error while stopping the session: ' + err.responseText);
    })
  });
});