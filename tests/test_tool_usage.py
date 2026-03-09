#!/usr/bin/env python3
"""
Test script to verify tool usage summary feature.
"""

import sys
sys.path.insert(0, 'src')

def test_tool_usage_in_base_agent():
    """Test that BaseAgent returns tool_usage_summary."""
    from arag.agent.base import BaseAgent
    
    # Check if the method exists
    assert hasattr(BaseAgent, '_calculate_tool_usage')
    print("✓ BaseAgent has _calculate_tool_usage method")
    
    # Test the method
    test_trajectory = [
        {"tool_name": "keyword_search"},
        {"tool_name": "read_chunk"},
        {"tool_name": "keyword_search"},
        {"tool_name": "semantic_search"},
        {"tool_name": "read_chunk"},
    ]
    
    # Create a mock instance to test the method
    class MockLLM:
        pass
    
    class MockTools:
        def get_all_schemas(self):
            return []
    
    agent = BaseAgent(
        llm_client=MockLLM(),
        tools=MockTools(),
    )
    
    usage = agent._calculate_tool_usage(test_trajectory)
    
    assert usage["keyword_search"] == 2
    assert usage["read_chunk"] == 2
    assert usage["semantic_search"] == 1
    print(f"✓ Tool usage calculation works: {usage}")

def test_tool_usage_in_memory_agent():
    """Test that MemoryAgent returns tool_usage_summary."""
    from arag.agent.memory_agent import MemoryAgent
    
    # Check if the method exists
    assert hasattr(MemoryAgent, '_calculate_tool_usage')
    print("✓ MemoryAgent has _calculate_tool_usage method")
    
    # Test the method
    test_trajectory = [
        {"tool_name": "memory_update"},
        {"tool_name": "memory_update"},
        {"tool_name": "memory_update"},
        {"tool_name": "final_answer"},
    ]
    
    # Create a mock instance
    class MockLLM:
        pass
    
    agent = MemoryAgent(
        llm_client=MockLLM(),
        context="test context",
    )
    
    usage = agent._calculate_tool_usage(test_trajectory)
    
    assert usage["memory_update"] == 3
    assert usage["final_answer"] == 1
    print(f"✓ Tool usage calculation works: {usage}")

def main():
    print("="*60)
    print("Tool Usage Summary Feature Test")
    print("="*60)
    
    try:
        test_tool_usage_in_base_agent()
        test_tool_usage_in_memory_agent()
        
        print("\n" + "="*60)
        print("✅ All tool usage tests passed!")
        print("="*60)
        print("\nNew feature added:")
        print("  - tool_usage_summary field in agent output")
        print("  - Shows count of each tool used")
        print("\nExample output:")
        print('  {')
        print('    "answer": "...",')
        print('    "tool_usage_summary": {')
        print('      "keyword_search": 2,')
        print('      "semantic_search": 1,')
        print('      "read_chunk": 3')
        print('    },')
        print('    ...')
        print('  }')
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
