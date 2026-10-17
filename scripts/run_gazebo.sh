gz sim warehouse_with_pallet.sdf &
sleep 10
wmctrl -r "Gazebo Sim" -b add,maximized_vert,maximized_horz

gz service -s /world/warehouse_world/control \
  --reqtype gz.msgs.WorldControl \
  --reptype gz.msgs.Boolean \
  --timeout 3000 \
  --req 'pause: false'

