all: main_window_ui.py resources_rc.py

.INTERMEDIATE: qtdesigner_sux

main_window_ui.py: main_window_ui.ui
	pyuic5 $< --import-from=. -o $@

resources_rc.py: resources.qrc res/*.png
	pyrcc5 $< -o $@
