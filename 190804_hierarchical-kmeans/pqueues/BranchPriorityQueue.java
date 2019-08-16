/*******************************************************************************
 * Copyright (C) 2017 Chang Wei Tan
 * 
 * This file is part of TSI.
 * 
 * TSI is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * 
 * TSI is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with TSI.  If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/
package pqueues;
import tree.TreeNode;

public class BranchPriorityQueue {
	/* Branch Priority Queue
	 * This is a min priority queue class for storing branches
	 * 
	 * Last modified: 14/01/2017
	 */
	private int lastElement;		// last element of the queue
	private int capacity;			// capacity of the queue
	private double[] distance;		// distance, used to sort the queue
	private TreeNode[] data;		// data to be stored
	
	private double topDistance;		// head of the queue
	private int topDataIndex;		// first data index
	private TreeNode topData;		// first data of the queue
	
	public BranchPriorityQueue(final int c) {
		capacity = c+1;						
		distance = new double[capacity];	
		data = new TreeNode[capacity];
		lastElement = 0;
	}
	
	public final void insert(final double d, final TreeNode t) {		
		if (lastElement < capacity-1) {
			lastElement++;
		} else {
			throw new RuntimeException("Queue is full!");
		}
		
		// insert to the last element
		distance[lastElement] = d;
		data[lastElement] = t;
		
		// go through the queue to make sure that the top is minimum
		int B = lastElement;
		int A = B/2;
		while (B > 1 && distance[A] > distance[B]) {
			swap(B, A);
			B = A;
			A = B/2;
		}
	}
	
	public final TreeNode pop() {
		if (lastElement == 0)
			throw new RuntimeException("Queue is empty!");
		
		// get dequeued data
		topDistance = distance[1];
		topData = data[1];
		
		// swap last and first data
		swap(1, lastElement);
		
		// remove last data
		distance[lastElement] = -1;
		data[lastElement] = null;
		
		lastElement--;
		
		// go through the queue to make sure that the top is minimum
		int A = 1;
		int B = 2*A;
		while(B <= lastElement) {
			if (B+1 <= lastElement && distance[B+1] < distance[B]) {
				B++;
			}
			
			if (distance[A] <= distance[B]) {
				break;
			}
			
			swap(A, B);
			A = B;
			B = 2*A;
		}
		
		return topData;
	}
	
	public final double firstDistance() {
		if (isEmpty())
			return Double.POSITIVE_INFINITY;
		else
			return distance[1];
	} 
	
	public final TreeNode popData() {
		return topData;
	}
	
	public final TreeNode firstData() {
		return data[1];
	}
	
	public final double popDistance() {
		return topDistance;
	}
	
	public final int popDataIndex() {
		return topDataIndex;
	}
	
	public final boolean isFull() {
		return lastElement == capacity-1;
	}
	
	public final boolean isEmpty() {
		return lastElement == 0;
	}
	
	public final int sizeOf() {
		return lastElement;
	}
	
	private final void swap(final int A, final int B) {
		final double tempDist = distance[A];
		final TreeNode tempData = data[A];
		
		distance[A] = distance[B];
		data[A] = data[B];
		
		distance[B] = tempDist;
		data[B] = tempData;
	}
}
