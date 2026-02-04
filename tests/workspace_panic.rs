#[test]
fn test_workspace_resize_panic_safety() {
    use reagle::utils::workspace::ThreadWorkspace;

    // Create a workspace with 0 states (should result in empty buffers)
    let mut ws = ThreadWorkspace::new(64, 0);
    assert_eq!(ws.fwd.len(), 0);

    // Resize to a valid size. This should NOT panic or result in 0-sized buffers.
    ws.resize_for_states(100);
    
    assert!(ws.fwd.len() > 0, "Forward buffer should be allocated");
    assert!(ws.fwd.len() >= 100, "Forward buffer should hold at least 100 states");
    
    // Resize again
    ws.resize_for_states(200);
    assert!(ws.fwd.len() >= 200, "Forward buffer should hold at least 200 states");
}
